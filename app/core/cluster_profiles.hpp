#ifndef SIRIUS_APP_CLUSTER_PROFILES_HPP
#define SIRIUS_APP_CLUSTER_PROFILES_HPP

// The cluster profiles as the settings file keeps them (sirius-app.toml,
// core/settings_toml.hpp): one table per cluster, and the one in use.
//
//   [cluster]
//   current = "mycluster"
//
//   [cluster.mycluster]
//   host = "mycluster"                  # ssh destination (~/.ssh/config alias or user@host)
//   image = "/path/sirius-worker.sif"   # the worker image (required to start a worker)
//   images = [...]                      # images used before (the dropdown)
//   binds = ["/data", "/scratch/me"]    # data folders the image sees
//   bind_sets = [...]                   # data folders used before (the dropdown)
//   checkout = "~/sirius"
//   launcher = "apptainer"
//   python_path = ""                    # extra PYTHONPATH inside the image
//   engine = true
//   engine_builds = "/path/sirius-engines"   # <commit>/bin/sirius-cli + BUILD.json
//   engine_bin = ""                     # an engine named outright
//   cache = ""                          # the node cache folder
//   def_file = ""                       # what a new image is built from
//   models = ""                         # the models folder Models… lists (Foundation step)
//   [cluster.mycluster.job]             # the job asked for
//   partition = "gpu"  account = "lab"  qos = "normal"  time = "01:00:00"
//   gpus = 1  cpus = 8  mem = "64G"
//   [[cluster.mycluster.partitions]]    # what the dropdowns offer
//   name = "gpu"  default = true  accounts = ["lab"]  qos = ["normal"]
//   max_time = "3-00:00:00"  times = [...]  gpus = 1  cpus = 8  mem = "64G"
//   max_gpus = 4  max_cpus = 64  max_mem = "500G"
//
// docs/clusters.example.toml is a commented example. Nothing here is a
// secret: passwords are asked by ssh and never kept, the worker's token is
// made anew for each worker.

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/cluster.hpp"

namespace sirius::app::cluster {

    // The settings' keys: "cluster/current", "cluster/<name>" (a profile),
    // and the single profile of before ("cluster/profile"), read once.
    inline constexpr const char* kCurrentKey = "cluster/current";
    inline constexpr const char* kLegacyProfileKey = "cluster/profile";
    inline constexpr const char* kRecentFoldersKey = "cluster/recentFolders";
    std::string profileKey(const std::string& name);
    // Names the [cluster] table has for itself: no profile may be called so.
    bool isReservedName(const std::string& name);

    nlohmann::json toJson(const PartitionChoice& c);
    PartitionChoice partitionChoiceFromJson(const nlohmann::json& j);

    // Every cluster profile, one per cluster, and the one in use.
    struct ProfileBook {
        std::vector<Profile> profiles;   // in name order
        std::string current;             // a name in `profiles`; "" = none

        // The profiles in the settings (`flat`: "group/name" keys, those
        // under "cluster/" at least). Without any, the single profile of
        // before ("cluster/profile") becomes one named after its host, every
        // value kept (`migrated` says so).
        static ProfileBook fromSettings(const nlohmann::json& flat, bool* migrated = nullptr);
        // The keys to write: "cluster/<name>" for each profile and
        // "cluster/current". Keys of profiles no longer there are the
        // caller's to remove (staleKeys).
        std::map<std::string, nlohmann::json> toSettings() const;
        // The "cluster/<name>" keys of `flat` that hold a profile this book has not.
        std::vector<std::string> staleKeys(const nlohmann::json& flat) const;

        Profile* find(const std::string& name);
        const Profile* find(const std::string& name) const;
        // The current profile; an empty one (no name) when there is none.
        Profile currentProfile() const;
        // `base` when no profile has that name (and it may be one), else
        // "base 2", "base 3", ...
        std::string uniqueName(const std::string& base) const;
        // Stores `p` under its displayName() (replacing one of that name) and makes it current.
        void put(Profile p);
        // false when `from` is not there or `to` is not a name it may take.
        bool rename(const std::string& from, const std::string& to);
        // Removes it; the current one becomes the first left.
        bool remove(const std::string& name);
        // Why `name` cannot name a profile ("" when it can): empty, a slash,
        // one of [cluster]'s own keys, or taken (by another than `except`).
        std::string nameProblem(const std::string& name, const std::string& except = {}) const;
    };

    // --- the dropdowns -----------------------------------------------------------

    // A partition as the cluster reports it, as a choice to keep ("Add to
    // my settings"): its accounts and QoS from the user's associations, its
    // time limit, a node's GPUs, CPUs and memory as the limits.
    PartitionChoice choiceFromCluster(const ClusterInfo& info, const std::string& partition);
    // Picking a partition from the profile's choices: the account and QoS
    // (the choice's first, unless the profile's are among its), the
    // resources it sets, the time and resources within its limits. Returns
    // what changed, in words.
    std::vector<std::string> applyChoice(Profile& p, const PartitionChoice& c);
    // The time limits a dropdown offers: the choice's own, else the usual ones
    // (30 min .. 7 days) up to the partition's and the QoS's limit (`limit`,
    // Slurm's form, "" none), with `current` among them.
    std::vector<std::string> timeChoices(const PartitionChoice* c, const std::string& limit, const std::string& current);

    // --- sharing a profile ---------------------------------------------------------

    // One profile as a small TOML file ([cluster.<name>]), to hand to a colleague.
    std::string exportProfile(const Profile& p);
    // The profiles of such a file (or of a whole settings file); throws
    // std::runtime_error with the line and column when it does not read,
    // or names no profile.
    std::vector<Profile> importProfiles(const std::string& text);

    // --- the settings editor's checks --------------------------------------------------

    struct SettingsProblem {
        std::vector<std::string> path;   // {"cluster", "<name>", "image"}: where, for settings_toml::position
        std::string message;             // what is wrong and what to do, in words
        bool error = true;               // false: said, but saving goes ahead
        int line = 0, column = 0;        // checkSettingsText: 1-based; 0 unknown
    };
    // The [cluster] tables of `flat`: types, the keys a profile knows, a host
    // and an image, partitions with names, times and sizes Slurm reads.
    std::vector<SettingsProblem> checkClusterSettings(const nlohmann::json& flat);
    // A settings file's text: TOML (the error where toml++ stops), then the
    // cluster checks with where each is in the text.
    std::vector<SettingsProblem> checkSettingsText(const std::string& text);

} // namespace sirius::app::cluster

#endif // SIRIUS_APP_CLUSTER_PROFILES_HPP
