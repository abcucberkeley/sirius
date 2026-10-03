#ifndef SIRIUS_APP_SETTINGS_TOML_HPP
#define SIRIUS_APP_SETTINGS_TOML_HPP

// The settings file's TOML (sirius-app.toml) and the application's settings
// as it holds them in memory: one JSON object keyed by "group/name"
// ("worker/python", "cluster/current", "cluster/<profile>" ...).
//
//   "group/name": value   <->   [group]
//                                name = value
//
// A key without a slash stays at the top of the file. Objects become
// tables ([cluster.<profile>]), arrays of objects arrays of tables
// ([[cluster.<profile>.partitions]]); integers stay integers and floats
// floats; a null is left out (a key that is not there reads as null). A
// string that is not UTF-8 (a Linux file name) is written with U+FFFD, as
// the JSON file of before did. Keys under "secrets/" are never written.
//
// toml++ (cmake/Dependencies.cmake) parses and writes it; it keeps no
// comments, so the file is written with a header and a line above each
// group of SIRIUS's own saying what it is, and says that comments added by
// hand are not kept.

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace sirius::app::settings_toml {

    // The settings as the file's text, the header and the groups' comments included.
    // `header`: the file's header comment first (not for a profile exported alone).
    std::string toToml(const nlohmann::json& flat, bool header = true);

    struct ParseResult {
        bool ok = false;
        nlohmann::json flat = nlohmann::json::object();
        std::string error;         // what toml++ said, without the position
        int line = 0, column = 0;  // 1-based; 0 = unknown
    };
    // The file's text as the settings; on an error, nothing (`flat` empty) and where.
    ParseResult fromToml(const std::string& text);

    // Where `path` (table names, keys, array indices as numbers: {"cluster",
    // "fiona", "partitions", "0", "name"}) is in `text`, 1-based line and
    // column; the nearest part of it that is there when the rest is not,
    // nullopt when the text does not parse or nothing of it is there.
    std::optional<std::pair<int, int>> position(const std::string& text, const std::vector<std::string>& path);

    // The flat key and the path inside its value of a path as position()
    // takes it: {"cluster", "fiona", "image"} -> ("cluster/fiona", {"image"}).
    std::pair<std::string, std::vector<std::string>> splitPath(const std::vector<std::string>& path);

    // `text` with every invalid UTF-8 sequence replaced by U+FFFD.
    std::string validUtf8(const std::string& text);

} // namespace sirius::app::settings_toml

#endif // SIRIUS_APP_SETTINGS_TOML_HPP
