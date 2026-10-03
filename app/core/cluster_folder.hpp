#ifndef SIRIUS_APP_CLUSTER_FOLDER_HPP
#define SIRIUS_APP_CLUSTER_FOLDER_HPP

// A folder of TIFF stacks on the cluster made into one dataset, as "Open
// folder as dataset" makes one of a folder on this computer -- the same
// filename pattern, the same manifest (core/manifest.hpp), the same dataset:
//
//   list    the folder's TIFF names over the session's command channel (a
//           names-only listing, capped; its sirius-dataset.toml read along
//           when it has one), in the order tiffNamesOf puts them
//   build   manifestFromNames over those names, the shape of the first file
//           of each tile asked of the engine on the node (dataset_info,
//           through the cluster:// opener core/remote_source.hpp installs)
//   keep    the manifest written to ~/.sirius/manifests on the cluster, with
//           files_folder naming the data folder: nothing is ever written into
//           the data folder (it is often read-only, and always someone's
//           data). The Load step's Source is then
//           cluster://<host>/<home>/.sirius/manifests/<name>.toml, which the
//           engine's DatasetService opens like any manifest, and which a
//           pipeline file keeps.

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/cluster.hpp"
#include "core/manifest.hpp"

namespace sirius::app {

    // Names listed at most: a folder of an acquisition holds thousands.
    inline constexpr int kClusterFolderCap = 200000;

    struct ClusterFolder {
        std::string host;                          // the ssh host its cluster:// paths name
        std::string path;                          // absolute, as the cluster resolved it
        std::string home;                          // the user's $HOME there
        std::vector<std::string> tiffs;            // TIFF names, in tiffNamesOf's order
        std::size_t others = 0;                    // files that are not TIFFs
        bool truncated = false;                    // more TIFFs than the cap: the rest are not listed
        bool store = false;                        // a zarr / N5 store, not a folder of stacks
        std::optional<DatasetManifest> existing;   // its sirius-dataset.toml, when it has a readable one
        std::string existingError;                 // ... or why that one could not be read

        // "cluster://<host>/<path>"
        std::string clusterPath() const;
        // "cluster://<host>/<path>/<name>"
        std::string clusterPathOf(const std::string& name) const;
    };

    // The script that lists `path` (exposed for tests), and its answer.
    std::string clusterFolderScript(const std::string& path, int cap);
    ClusterFolder parseClusterFolder(const std::string& output);

    // Lists `path` (a cluster path: "/data/acq", "~/acq") through `session`;
    // throws ssh::SshError (not logged in, a folder that cannot be read).
    ClusterFolder listClusterFolder(cluster::Session& session, const std::string& host, const std::string& path,
                                    int cap = kClusterFolderCap);

    // The manifest of `folder` by `rule`, the shapes from the engine on the
    // node (probeDataset of each tile's first file). Throws as
    // manifestFromNames does, and with the opener's words when no worker is
    // connected. files_folder names the folder.
    DatasetManifest manifestFromClusterFolder(const ClusterFolder& folder, const FilenameRule& rule,
                                              std::vector<std::string>* unmatched = nullptr);

    // The manifest's file name in ~/.sirius/manifests: the folder's name and
    // a hash of the host and path ("acq-1f3a...e2.toml"), so a folder keeps
    // its manifest and two folders of one name do not share one.
    std::string clusterManifestName(const std::string& host, const std::string& folder);
    // The script that writes `text` there (exposed for tests).
    std::string writeClusterManifestScript(const std::string& text, const std::string& fileName);
    // Writes `manifest` (files_folder set to the folder) to the cluster's
    // ~/.sirius/manifests; returns its cluster:// path, which the Load step
    // opens. Throws ssh::SshError when it cannot be written.
    std::string writeClusterManifest(cluster::Session& session, const ClusterFolder& folder, DatasetManifest manifest);

} // namespace sirius::app

#endif // SIRIUS_APP_CLUSTER_FOLDER_HPP
