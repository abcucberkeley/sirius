#ifndef SIRIUS_APP_MANIFEST_HPP
#define SIRIUS_APP_MANIFEST_HPP

// Multi-file datasets. A folder of TIFF stacks -- one file per channel, time
// point and tile -- becomes one dataset through a manifest
// (`sirius-dataset.toml` in the folder) that maps every file to its channel,
// time point and tile and records voxel size, channel names and tile
// positions. The manifest is written once, typically from a filename
// pattern with named groups (FilenameRule), and read on every later open.

#include <array>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "core/dataset.hpp"

namespace sirius::app {

    struct ManifestFile {
        std::string path;        // relative to the folder (or absolute)
        std::string channel;     // channel name (matches ChannelInfo::label) or index as text
        Index t = 0;             // time point
        std::string tile;        // TileInfo::name; empty = the only tile
    };

    struct DatasetManifest {
        static constexpr const char* kFileName = "sirius-dataset.toml";

        std::string name;
        std::array<double, 3> voxelUm{0.1, 0.1, 0.2};   // x, y, z
        double frameIntervalS = 0.0;
        std::vector<ChannelInfo> channels;              // in channel-index order
        std::vector<TileInfo> tiles;                    // at least one
        std::vector<ManifestFile> files;
        SimLayout sim;
        std::string acquisition;
        std::string pattern;                            // FilenameRule::pattern that produced it, for reference
        // TIFF directory when this file is stored somewhere else. Empty: the
        // directory that holds the manifest (the usual in-folder case).
        std::string filesFolder;

        // Where relative file paths are resolved, given the path load() used.
        std::filesystem::path filesRoot(const std::filesystem::path& loadedFrom) const;

        Index channelIndex(const std::string& channel) const noexcept;   // -1 when unknown
        Index tileIndex(const std::string& tile) const noexcept;         // -1 when unknown; "" -> 0
        Index timePoints() const noexcept;                              // max t + 1
        const ManifestFile* file(Index tile, Index channel, Index t) const noexcept;

        nlohmann::json toJson() const;
        static DatasetManifest fromJson(const nlohmann::json& j);
        void save(const std::filesystem::path& path) const;              // TOML
        static DatasetManifest load(const std::filesystem::path& path);
        // Problems that make the manifest unusable for `folder`: missing files,
        // unknown channels / tiles, a (tile, channel, t) with no file. Empty = fine.
        std::vector<std::string> validate(const std::filesystem::path& folder) const;
    };

    // How filenames map to channel / time / tile. `pattern` is a regular
    // expression (ECMAScript) matched against the file name (not the path)
    // with named groups, written (?P<name>...) or (?<name>...):
    //   channel | c        channel name or index
    //   t | time           time point index
    //   tile               tile name (else built from x / y / z)
    //   x | col, y | row, z   grid indices (Positions::GridIndex) or stage
    //                      coordinates in micrometres (Positions::Microns)
    // Unmatched files are ignored and reported.
    struct FilenameRule {
        std::string pattern;
        enum class Positions { None,
                               GridIndex,
                               Microns };
        Positions positions = Positions::GridIndex;
        double overlapFraction = 0.10;       // GridIndex: neighbouring tiles overlap by this fraction
        std::array<double, 3> voxelUm{0.1, 0.1, 0.2};
        double frameIntervalS = 0.0;
        SimLayout sim;
        std::string acquisition;
        // Optional per-channel-token names / wavelengths ("488" -> {label "GFP", 488 nm, colour}).
        std::map<std::string, ChannelInfo> channelInfo;
    };

    struct FilenameMatch {
        std::string file;
        bool matched = false;
        std::map<std::string, std::string> groups;   // named groups that matched
    };

    // Preview: which files the pattern matches and what it extracts. Throws
    // std::invalid_argument for a pattern that does not compile.
    std::vector<FilenameMatch> matchFilenames(const std::vector<std::string>& names, const std::string& pattern);
    // Named-group support for std::regex: strips the names, returning the
    // plain pattern and the group order (name of group 1, 2, ...).
    std::string plainPattern(const std::string& pattern, std::vector<std::string>* groupNames);

    // Where a manifest entry lives: relative paths are taken from the folder.
    std::filesystem::path manifestFilePath(const std::filesystem::path& folder, const ManifestFile& file);

    // Build a manifest for the TIFF files of `folder` (non-recursive) with the
    // rule; tile shapes are probed from one file per tile (all must agree).
    // Files the pattern does not match are listed in `unmatched`.
    DatasetManifest manifestFromFolder(const std::filesystem::path& folder, const FilenameRule& rule,
                                       std::vector<std::string>* unmatched = nullptr);

    // The simple case, without a pattern to write: every TIFF in the folder is
    // one time point of one stack, in the order a person reads the names --
    // "f2" before "f10", which plain alphabetical order gets wrong. One
    // channel, one tile. This is what a time series saved a frame per file
    // looks like, and it is the folder layout that needs no describing.
    DatasetManifest manifestOfOneStack(const std::filesystem::path& folder);

    // File names in that reading order, for the dialog to show and count.
    std::vector<std::string> tiffNamesInOrder(const std::filesystem::path& folder);

} // namespace sirius::app

#endif // SIRIUS_APP_MANIFEST_HPP
