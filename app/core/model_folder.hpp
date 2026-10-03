#ifndef SIRIUS_APP_MODEL_FOLDER_HPP
#define SIRIUS_APP_MODEL_FOLDER_HPP

// A trained model as the Foundation step takes it: a self-contained folder
// (latents scripts/export_model.py writes it), run by the Python worker
// without the latents package:
//
//   <models>/<name>/<version>/
//       model.py              the model's API: load(folder, device) -> Model
//       model.json            format "latents-model/1": tasks, the input
//                             contract, the decode rule, provenance
//       weights.safetensors   README.md   _lib/
//
// What the application needs of one is in model.json, plain JSON: the tasks
// it offers (the step's Task choice), its name and version, a description,
// the voxel size it was trained at. A folder on this computer is read here;
// one on the cluster (cluster://host/path) is described by the worker there
// (model_info), in the same fields.
//
// The single-file bundles of before (.ltb) needed the latents package to
// load. They are refused with what to do instead.

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sirius::app {

    struct ModelFolderFacts {
        std::string path;          // the folder (as given; a model.json path names its folder)
        std::string name, version;
        std::string description;   // model.json's notes, else README.md's first paragraph
        std::vector<std::string> tasks;   // "segment", "prompt"
        bool promptable = false;   // "prompt" is among the tasks
        std::vector<double> voxelUm;      // (x, y, z), the application's order; empty = not said
        int channels = 1;
        std::string channelMerge;
        long long sizeBytes = -1;  // weights.safetensors; -1 = not known
        std::string error;         // a listing's entry that is not a usable model: why

        bool offers(const std::string& task) const;
        // "coat-sam-s2 v1 · segment, prompt"
        std::string title() const;
    };

    // "Segment objects" / kPromptTask: the step's Task labels, and the tasks they are.
    inline constexpr const char* kModelSegmentLabel = "Segment objects";
    std::string modelTaskOfLabel(const std::string& label);   // "segment" | "prompt"
    std::string modelTaskLabel(const std::string& task);      // "" for a task the step has no label for

    // The Task labels a model offers, in the step's order; both when nothing
    // is known about the model yet.
    std::vector<std::string> modelTaskChoices(const std::optional<ModelFolderFacts>& facts);

    bool isOldBundlePath(const std::string& path);   // a .ltb file
    std::string oldBundleMessage(const std::string& path);

    // A folder on this computer: its model.json read and checked (format
    // latents-model/1, a task list). nullopt with `error` set when it is not
    // one: a sentence that says what to do. Cached by the file's time stamp,
    // so a panel may ask every frame.
    std::optional<ModelFolderFacts> readModelFolder(const std::string& path, std::string* error = nullptr);

    // The same facts from the worker's model_info answer or one entry of its
    // list_bundles listing ({name, version, tasks, description, voxel_um, ...});
    // nullopt (with `error`) when it is not a model folder's.
    std::optional<ModelFolderFacts> modelFactsFromJson(const nlohmann::json& info, std::string* error = nullptr);

    // The models under each of `dirs`, as Models… lists them:
    // <dir>/<name>/<version>/model.json, <dir>/<name>/model.json, or `dir`
    // being a model folder itself; two levels, sorted by name. A folder whose
    // model.json does not read, and an old .ltb beside the models, are listed
    // with `error` set; a `dir` that is not there goes to `errors`.
    struct ModelListing {
        std::vector<ModelFolderFacts> models;
        std::vector<std::string> errors;
    };
    ModelListing listModelFolders(const std::vector<std::string>& dirs);
    // The worker's list_bundles answer ({models: [...], errors: [...]}), for a folder it sees.
    ModelListing modelListingFromJson(const nlohmann::json& reply);

} // namespace sirius::app

#endif // SIRIUS_APP_MODEL_FOLDER_HPP
