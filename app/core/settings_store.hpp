#ifndef SIRIUS_APP_SETTINGS_STORE_HPP
#define SIRIUS_APP_SETTINGS_STORE_HPP

// The application's persistent settings: one TOML file, held in memory as
// one JSON object keyed by "group/name" strings ("worker/python",
// "recent/datasets", "cluster/<profile>" ...), which the file has as
// [group] tables (core/settings_toml.hpp). The file is
// <config dir>/sirius/sirius-app.toml (%APPDATA% on Windows,
// $XDG_CONFIG_HOME or ~/.config elsewhere) unless main() moved it
// (--settings <dir>); Dear ImGui's own window layout (imgui.ini) sits beside
// it.
//
// Thread-safe: the hub token and the worker settings are read on run threads.
// A set() marks the store dirty and autosave() -- called once per frame by
// the application -- or save() writes it, atomically (a file beside it,
// renamed over it). The file is read again for each save and only the keys
// this process set or removed change in it, and reads pick up within a
// second what another instance wrote, so two instances running side by side
// keep each other's settings.
//
// A file that does not parse (a hand edit gone wrong) is never written
// over: the application starts with its defaults, loadError() says what is
// wrong and where, and saves wait until the file is fixed (or replaced
// through adoptText(), the settings editor's save).
//
// The JSON file of before (sirius-app.json, or the older sirius-imgui.json)
// is read when there is no TOML file yet; the first save writes its
// settings as sirius-app.toml and renames it to <name>.migrated. Its
// secrets/ entries (Windows' DPAPI blobs) go to secrets.json beside it,
// never into the TOML file.

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sirius::app {

    class Settings {
    public:
        // The one store of the process.
        static Settings& instance();

        // Keeps the settings (and imgui.ini) in `dir`; before the first read.
        void setDirectory(const std::string& dir);
        std::string directory() const;
        std::string filePath() const;     // sirius-app.toml
        std::string layoutPath() const;   // imgui.ini

        bool contains(const std::string& key) const;
        nlohmann::json value(const std::string& key, const nlohmann::json& def = nullptr) const;
        std::string getString(const std::string& key, const std::string& def = {}) const;
        int getInt(const std::string& key, int def = 0) const;
        double getDouble(const std::string& key, double def = 0.0) const;
        bool getBool(const std::string& key, bool def = false) const;
        std::vector<std::string> getStringList(const std::string& key) const;

        void set(const std::string& key, const nlohmann::json& value);
        void remove(const std::string& key);
        // Every key that starts with `prefix` ("cluster/").
        std::vector<std::string> keys(const std::string& prefix = {}) const;

        // Writes the file when something changed since the last save. False
        // when the file could not be written (or is there and does not
        // parse): the changes stay in memory and the next save() or
        // autosave() tries them again.
        bool save();
        // save() for the frame loop: after a failed write, the same changes
        // are tried again once a second rather than on every frame.
        void autosave();
        // set(), or remove() for nullopt, and save() in one, for a change the
        // caller must know the fate of (a secret). False when the file could
        // not be written, and the change is then taken back rather than kept
        // for a later save: the key reads as it did, and the file keeps the
        // value it has, which may be another instance's.
        bool commit(const std::string& key, const std::optional<nlohmann::json>& value);

        // What is wrong with the file as last read ("line 12, column 7:
        // expected '='"); "" when it read (or is not there).
        std::string loadError() const;
        // The settings editor's save: `text` (TOML) parsed and, when it
        // parses, written as it is (its comments too) and taken as the
        // settings at once -- this process's unwritten changes are dropped,
        // the text is what the user wants. False with `error` (and nothing
        // written) when it does not parse or cannot be written.
        bool adoptText(const std::string& text, std::string* error);

    private:
        Settings() = default;
        struct State;
        State& state() const;
        // Reads the file on first use, and again when another process wrote
        // it (looked at once a second); caller holds the mutex.
        void load() const;
        bool saveLocked();   // save(); caller holds the save mutex
    };

} // namespace sirius::app

#endif // SIRIUS_APP_SETTINGS_STORE_HPP
