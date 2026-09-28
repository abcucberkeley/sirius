#ifndef SIRIUS_IMGUI_SETTINGS_HPP
#define SIRIUS_IMGUI_SETTINGS_HPP

// The application's persistent settings: one JSON object in one file, keyed
// by "group/name" strings ("worker/python", "recent/datasets",
// "assistant/model" ...). The file is <config dir>/sirius/sirius-app.json unless
// main() moved it (--settings <dir>); Dear ImGui's own window layout
// (imgui.ini) sits beside it.
//
// Thread-safe: the hub token and the worker settings are read on run threads.
// A set() marks the store dirty and autosave() -- called once per frame by
// the application -- or save() writes it, atomically (a file beside it,
// renamed over it). The file is read again for each save and only the keys
// this process set or removed change in it, and reads pick up within a
// second what another instance wrote, so two instances running side by side
// keep each other's settings.

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sirius::app::gui {

    class Settings {
    public:
        // The one store of the process.
        static Settings& instance();

        // Keeps the settings (and imgui.ini) in `dir`; before the first read.
        void setDirectory(const std::string& dir);
        std::string directory() const;
        std::string filePath() const;
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
        // Every key that starts with `prefix` ("secrets/").
        std::vector<std::string> keys(const std::string& prefix = {}) const;

        // Writes the file when something changed since the last save. False
        // when the file could not be written: the changes stay in memory and
        // the next save() or autosave() tries them again.
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

    private:
        Settings() = default;
        struct State;
        State& state() const;
        // Reads the file on first use, and again when another process wrote
        // it (looked at once a second); caller holds the mutex.
        void load() const;
        bool saveLocked();   // save(); caller holds the save mutex
    };

    inline Settings& settings() { return Settings::instance(); }

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_SETTINGS_HPP
