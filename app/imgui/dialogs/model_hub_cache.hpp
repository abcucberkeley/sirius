#ifndef SIRIUS_IMGUI_MODEL_HUB_CACHE_HPP
#define SIRIUS_IMGUI_MODEL_HUB_CACHE_HPP

// The model cache of this machine, as the Python worker keeps it
// (app/python/sirius_worker/models.py): $SIRIUS_MODEL_CACHE or
// ~/.sirius/models, with one flat directory per Hugging Face repository
// (hf/<owner>--<repo>/<file>) and the user's own files under local/.
//
// The model hub reads it in this process rather than through the worker: the
// cache is a directory on this machine whichever backend computes, and the
// list of what was downloaded must not wait for (or fail with) a Python
// interpreter. The layout and the rules are the worker's, so a file fetched
// by one is found by the other.
//
// All paths are UTF-8. Nothing here touches the network.

#include <cstdint>
#include <string>
#include <vector>

namespace sirius::app::gui::modelhub {

    struct CachedModel {
        std::string spec;      // "hf:<repo>:<file>", or the path of a file under local/
        std::string path;
        std::string repo;      // "" for a local file
        std::string file;      // inside the repository, with forward slashes
        std::uint64_t bytes = 0;
    };

    struct Deleted {
        std::string path;                              // resolved
        std::uint64_t bytes = 0;                       // what that freed
        std::vector<std::string> removedDirectories;   // the empty ones the file left behind
    };

    // .pt, .pts, .pth, .onnx (any case): what SIRIUS can run.
    bool isModelFile(const std::string& name);

    // $SIRIUS_MODEL_CACHE (a leading ~ is the home directory), else ~/.sirius/models.
    std::string cacheDirectory();
    std::string repositoryDirectory(const std::string& repo);

    // Where a repository file is kept once downloaded. Throws
    // std::runtime_error for a name that reaches outside the repository's
    // directory (".." or a drive in it): the name comes from the network.
    std::string downloadTarget(const std::string& repo, const std::string& file);
    // The path of an already downloaded repository file, else "".
    std::string cachedPath(const std::string& repo, const std::string& file);

    // The model files in the cache, repositories first, each sorted by name.
    std::vector<CachedModel> listCachedModels();

    // Removes one model (a file, or a repository's directory) from the cache
    // and the directories it leaves empty. Only paths inside the cache are
    // touched; anything else throws std::runtime_error with the reason.
    Deleted deleteCachedModel(const std::string& path);

} // namespace sirius::app::gui::modelhub

#endif // SIRIUS_IMGUI_MODEL_HUB_CACHE_HPP
