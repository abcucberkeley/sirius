#ifndef SIRIUS_APP_TRAINING_EXPORT_HPP
#define SIRIUS_APP_TRAINING_EXPORT_HPP

// Turning a segmented volume into training data.
//
// The classical steps produce instance labels; a model wants those in the
// three shapes the literature trains on -- instance masks, bounding boxes and
// a semantic (class per voxel) mask -- written into a folder that accumulates
// one sample per export, with a class table and an index the loader can read.
//
// The four functions above exportTrainingData() are pure: they derive the
// boxes, the class table and the semantic volume from a LabelVolume without
// touching the disk, so they can be tested exactly and reused by the review
// table and the tool API.

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/array.hpp"
#include "core/dataset.hpp"
#include "core/export.hpp"
#include "core/labels.hpp"

namespace sirius::app {

    // Class names in a fixed order. Background is 0 and is not in `names`, so
    // the id of names[i] is i + 1: what a semantic mask stores.
    struct ClassTable {
        std::vector<std::string> names;

        int idOf(const std::string& name) const noexcept;   // 0 when not present
        const std::string& nameOf(int id) const noexcept;   // "background" for 0 / out of range
        std::size_t size() const noexcept { return names.size(); }
        bool empty() const noexcept { return names.empty(); }
    };

    // Every class of every object in every frame (LabelVolume::annotationOf,
    // so each frame's own), sorted; "object" for the labels nothing was said
    // about. The functions below name the class of a label the same way.
    ClassTable classTable(const LabelVolume& labels);

    // One object of one time point.
    struct LabelBox {
        std::uint32_t label = 0;
        int classId = 0;
        std::string className = "object";
        Index t = 0;
        std::array<Index, 6> bbox{};        // z0, z1, y0, y1, x0, x1, half open
        std::uint64_t voxels = 0;
        double cz = 0.0, cy = 0.0, cx = 0.0;   // centroid, voxels
        bool touchesBorder = false;

        Index dz() const noexcept { return bbox[1] - bbox[0]; }
        Index dy() const noexcept { return bbox[3] - bbox[2]; }
        Index dx() const noexcept { return bbox[5] - bbox[4]; }
    };

    // Scans the voxels of time point `t`; labels under `minVoxels` are left
    // out. Ordered by label id, so two runs agree.
    std::vector<LabelBox> boundingBoxes(const LabelVolume& labels, Index t, const ClassTable& classes,
                                        std::uint64_t minVoxels = 1);

    // The same per z plane: what a slice-wise detector trains on. A label
    // that a plane cuts in two still gives that plane one box.
    struct SliceBox {
        Index z = 0;
        std::uint32_t label = 0;
        int classId = 0;
        Index y0 = 0, y1 = 0, x0 = 0, x1 = 0;   // half open
        std::uint64_t pixels = 0;
    };
    std::vector<SliceBox> sliceBoxes(const LabelVolume& labels, Index t, const ClassTable& classes,
                                     std::uint64_t minPixels = 1);

    // Class id per voxel of time point `t`, (z, y, x), 0 = background.
    Buffer<std::uint8_t> semanticVolume(const LabelVolume& labels, Index t, const ClassTable& classes);

    // "0 cx cy w h" per line, normalised to the plane, as YOLO reads it
    // (class ids 0 based there, so one less than the semantic mask).
    std::string yoloText(const std::vector<SliceBox>& boxes, Index z, Index height, Index width);

    struct TrainingExportOptions {
        std::string directory;              // the dataset folder; it accumulates samples
        std::string sample;                 // subfolder name; a number is appended if taken
        bool image = true;                  // image.tif
        bool instances = true;              // instances.tif  (uint32, t * z pages)
        bool semantic = true;               // semantic.tif   (uint8 class ids)
        bool boxes = true;                  // boxes.json     (3D and per plane)
        bool slices = false;                // slices/: one 8 bit plane + one YOLO txt each
        bool overwrite = false;             // reuse the sample folder instead of numbering
        std::uint64_t minVoxels = 1;        // objects smaller than this are not training data
        PixelType imageDtype = PixelType::UInt16;
        ExportScaling scaling = ExportScaling::Percentile;
        double percentileLo = 0.1, percentileHi = 99.9;
        double rangeLo = 0.0, rangeHi = 1.0;
        std::string pipelineToml;           // provenance: how the labels were made
        nlohmann::json provenance;          // dataset path, step, parameters
    };

    struct TrainingExportResult {
        std::filesystem::path directory;    // the sample folder
        std::vector<std::string> files;     // relative to the dataset folder
        std::uint64_t objects = 0;
        std::uint64_t sliceObjects = 0;
        std::size_t classes = 0;            // in this sample; the ids index the dataset's classes.txt
        Index frames = 0;
        std::uint64_t bytes = 0;
    };

    // Empty when the options are consistent, otherwise the problem.
    std::string validateTrainingExport(const TrainingExportOptions& o, const LabelVolume& labels);
    // The same, and the image written beside the labels (image.tif, the
    // slice planes) has to be of the same (t, z, y, x) as they are: an image
    // of another grid would be a sample whose masks do not fit its image.
    std::string validateTrainingExport(const TrainingExportOptions& o, const LabelVolume& labels, const Dims5& imageDims);

    TrainingExportResult exportTrainingData(const Array5& array, const DatasetMeta& meta, const LabelVolume& labels,
                                            const TrainingExportOptions& options,
                                            const std::function<void(double, const std::string&)>& progress = {},
                                            const std::function<bool()>& cancelled = {});

} // namespace sirius::app

#endif // SIRIUS_APP_TRAINING_EXPORT_HPP
