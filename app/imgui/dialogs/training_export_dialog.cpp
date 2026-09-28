// File ▸ Export training data…: turns the labels of one step into a dataset
// folder -- instance masks, a semantic mask, bounding boxes, optionally one
// image and one YOLO file per plane -- and appends the sample to the index so
// many exports accumulate into one training set.

#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include "core/training_export.hpp"
#include "imgui/dialogs/export_dialog_support.hpp"
#include "imgui/platform.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {

        using namespace dialog_support;
        using theme::px;

        const PixelType kImageTypes[] = {PixelType::UInt8, PixelType::UInt16, PixelType::Float32};
        const ExportScaling kScalings[] = {ExportScaling::Percentile, ExportScaling::MinMax, ExportScaling::Cast};

        class TrainingExportDialog : public Dialog {
        public:
            TrainingExportDialog(App& app, std::function<void(int, const TrainingExportOptions&)> accepted)
                : accepted_(std::move(accepted)) {
                const Workbench& wb = app.wb();
                chosenStep(wb);   // no step chosen yet: the viewed one
                // defaults from the dataset: the folder beside it, the step as the name
                const DatasetMeta& ds = wb.dataset();
                std::string dir = ds.sourcePath.empty() ? std::string() : parentPath(ds.sourcePath);
                if (dir.empty() || dir == ".") dir = platform::homeDirectory();
#ifdef _WIN32
                dir = replaceAll(dir, "\\", "/");   // forward slashes throughout
#endif
                directory_ = dir + "/training-data";
                sample_ = ds.name.empty() ? std::string("sample") : ds.name;
            }

            std::string title() const override { return "Export training data"; }
            ImVec2 size() const override { return ImVec2(620, 0); }

            void draw(App& app) override {
                const Workbench& wb = app.wb();
                const Pipeline& p = wb.pipeline();
                int step = chosenStep(wb);
                const std::shared_ptr<const StepOutput> out = wb.output(step);
                const LabelVolume* labels = out ? out->labels.get() : nullptr;
                const bool labelled = labels != nullptr && !labels->empty();

                const Spacing spacing(8, 12);
                widgets::vspace(2);
                note("Writes one sample folder per export — instance masks, a semantic mask and "
                     "bounding boxes — and appends it to index.jsonl, so a folder collects the "
                     "output of many runs into one training set.");

                {
                    std::vector<std::string> steps;
                    for (int s = 0; s < p.size(); ++s) {
                        std::string label = Step::number(s) + " " + p.at(s).name;
                        const auto o = wb.output(s);
                        if (!o) label += "  (not computed)";
                        else if (!o->labels || o->labels->empty()) label += "  (no labels)";
                        else if (!wb.outputFresh(s)) label += "  (out of date)";
                        steps.push_back(std::move(label));
                    }
                    const Field f("Labels from step");
                    if (widgets::combo("##step", &step, steps)) choose(p, step);
                }
                // The sample keeps the pipeline as it is now as its provenance, and
                // after a parameter edit or an undo that is not the one that made
                // these labels.
                if (labelled && !wb.outputFresh(step))
                    note("The parameters changed since step " + Step::number(step) +
                             " was computed: these labels come from the "
                             "earlier parameters, and the sample's provenance records the current pipeline, "
                             "which did not make them. Run the step again for a matching record.",
                         theme::kAccentText);

                {
                    const Field f("Dataset folder");
                    const float browseW = buttonWidth("Browse…", widgets::ButtonKind::Ghost);
                    widgets::FieldOpts fo;
                    fo.width = design(std::max(px(60), ImGui::GetContentRegionAvail().x - browseW - px(8)));
                    fo.hint = "dataset folder";
                    widgets::inputText("##directory", &directory_, fo);
                    ImGui::SameLine(0.0f, px(8));
                    widgets::ButtonOpts b;
                    b.kind = widgets::ButtonKind::Ghost;
                    if (widgets::button("Browse…", b)) {
                        const std::string chosen = platform::pickFolderDialog("Training dataset folder", trimmed(directory_));
                        if (!chosen.empty()) directory_ = chosen;
                    }
                }
                {
                    const Field f("Sample");
                    widgets::FieldOpts fo;
                    fo.hint = "sample name";
                    widgets::inputText("##sample", &sample_, fo);
                }

                {
                    const Field f("Write");
                    const Spacing checks(8, 8);
                    widgets::checkbox("Image (image.tif)", &image_);
                    widgets::checkbox("Instance masks (instances.tif, one id per object)", &instances_);
                    widgets::checkbox("Semantic mask (semantic.tif, one id per class)", &semantic_);
                    widgets::checkbox("Bounding boxes (boxes.json, 3D and per plane)", &boxes_);
                    widgets::checkbox("2D slices (slices/: one 8-bit plane and one YOLO file each)", &slices_);
                }

                {
                    const float w = columnWidth(3, 10);
                    widgets::FieldOpts fo;
                    fo.width = design(w);
                    {
                        const Field f("Smallest object", "voxels");
                        widgets::inputInt("##minVoxels", &minVoxels_, 1, 1000000, 1, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f("Image pixel type");
                        std::vector<std::string> names;
                        for (PixelType t : kImageTypes) names.emplace_back(toString(t));
                        widgets::FieldOpts dt = fo;
                        dt.enabled = image_ || slices_;
                        widgets::combo("##dtype", &dtype_, names, dt);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f("Image scaling");
                        widgets::FieldOpts sc;
                        sc.enabled = image_;
                        widgets::combo("##scaling", &scaling_, {"rescale percentiles", "rescale min – max", "cast (no rescale)"}, sc);
                    }
                }

                const std::string summary = labelled ? summaryOf(out, *labels) : std::string();
                if (!summary.empty()) note(summary);

                std::string problem;
                if (!out) problem = "step " + Step::number(step) + " has not been computed yet; run it first";
                else if (!labelled) problem = "step " + Step::number(step) + " produced no labels; segment first";
                else problem = validateTrainingExport(options(), *labels);
                if (!problem.empty()) note(problem, theme::kAccentText);
                // A run or a task (a load, another export) holds the worker and
                // the export task would be refused: Export waits, with the
                // dialog kept open.
                const bool busy = app.bridge().busy();
                if (busy) note("A run or a task is in progress: Export is available again when it has finished.", theme::kAccentText);

                widgets::vspace(2);
                switch (actionRow("Export", problem.empty() && !busy)) {
                    case Action::Cancel: close(); break;
                    case Action::Accept:
                        if (accepted_) accepted_(step, options());
                        close();
                        break;
                    case Action::None: break;
                }
            }

        private:
            // The index of the chosen step, which is held by its id: the
            // assistant's tool calls run between frames and can add, remove
            // or move steps while the dialog is open, and an index kept from
            // the frame before then named another step, whose labels the
            // sample took and whose name its provenance recorded. A replaced
            // pipeline (the example, a dropped .sirius.toml) keeps the ids
            // its steps bring, so the id can come back as another step; no
            // edit changes a step's kind, so a step of another kind under it
            // is the replacement's, and is gone as the chosen one. (The name
            // is no test: a rename keeps the step.) A gone step's place is
            // taken by the viewed one. A replacement that puts a step of the
            // same kind under the id keeps it chosen; the old outputs went
            // with the old pipeline, so it has no labels until it runs.
            int chosenStep(const Workbench& wb) {
                const Pipeline& p = wb.pipeline();
                int step = p.indexOf(stepId_);
                if (step < 0 || p.at(step).kind != stepKind_) {
                    step = std::clamp(wb.viewedIndex(), 0, std::max(0, p.size() - 1));
                    choose(p, step);
                }
                return step;
            }

            void choose(const Pipeline& p, int step) {
                const bool known = step >= 0 && step < p.size();
                stepId_ = known ? p.at(step).id : StepId{0};
                stepKind_ = known ? p.at(step).kind : std::string();
            }

            std::string summaryOf(const std::shared_ptr<const StepOutput>& out, const LabelVolume& labels) {
                const std::uint64_t minVoxels = static_cast<std::uint64_t>(minVoxels_);
                if (out != countedOut_ || stepId_ != countedStepId_ || minVoxels != countedMin_) {
                    const ClassTable classes = classTable(labels);
                    std::uint64_t objects = 0;
                    for (Index t = 0; t < labels.t(); ++t) objects += boundingBoxes(labels, t, classes, minVoxels).size();
                    countedSummary_ = format("%llu object%s over %lld time point%s, %llu class%s.", static_cast<unsigned long long>(objects),
                                             objects == 1 ? "" : "s", static_cast<long long>(labels.t()), labels.t() == 1 ? "" : "s",
                                             static_cast<unsigned long long>(classes.size()), classes.size() == 1 ? "" : "es");
                    countedOut_ = out;
                    countedStepId_ = stepId_;
                    countedMin_ = minVoxels;
                }
                std::string summary = countedSummary_;
                if (slices_)
                    summary += format(" The slice output writes %lld plane files.", static_cast<long long>(labels.t() * labels.z() * 2));
                return summary;
            }

            TrainingExportOptions options() const {
                TrainingExportOptions o;
                o.directory = trimmed(directory_);
                o.sample = trimmed(sample_);
                if (o.sample.empty()) o.sample = "sample";
                o.image = image_;
                o.instances = instances_;
                o.semantic = semantic_;
                o.boxes = boxes_;
                o.slices = slices_;
                o.minVoxels = static_cast<std::uint64_t>(minVoxels_);
                o.imageDtype = kImageTypes[std::clamp(dtype_, 0, 2)];
                o.scaling = kScalings[std::clamp(scaling_, 0, 2)];
                return o;
            }

            std::function<void(int, const TrainingExportOptions&)> accepted_;
            StepId stepId_ = 0;
            std::string stepKind_;   // the chosen step's kind, which tells it from a replacement's under its id
            std::string directory_;
            std::string sample_;
            bool image_ = true, instances_ = true, semantic_ = true, boxes_ = true, slices_ = false;
            std::int64_t minVoxels_ = 1;
            int dtype_ = 1;     // uint16
            int scaling_ = 0;   // percentiles

            // boundingBoxes() walks every voxel of every time point. draw()
            // runs every frame, so the count is kept until the step or the
            // size filter moves.
            std::shared_ptr<const StepOutput> countedOut_;
            StepId countedStepId_ = 0;
            std::uint64_t countedMin_ = 0;
            std::string countedSummary_;
        };

    } // namespace

    std::shared_ptr<Dialog> makeTrainingExportDialog(App& app,
                                                     std::function<void(int stepIndex, const TrainingExportOptions& options)> accepted) {
        return std::make_shared<TrainingExportDialog>(app, std::move(accepted));
    }

} // namespace sirius::app::gui
