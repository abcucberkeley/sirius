#ifndef SIRIUS_IMGUI_SETTINGS_HPP
#define SIRIUS_IMGUI_SETTINGS_HPP

// The application's persistent settings: one JSON object in one file, keyed
// by "group/name" strings ("worker/python", "recent/datasets",
// "assistant/model" ...). The file is <config dir>/sirius/sirius-app.json unless
// main() moved it (--settings <dir>); Dear ImGui's own window layout
// (imgui.ini) sits beside it.
//
// Thread-safe: the hub token and the worker settings are read on run threads.
// A set() marks the store dirty and save() -- called once per frame by the
// application and at exit -- writes it, atomically (a file beside it, renamed
// over it).

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

        // Writes the file when something changed since the last save.
        void save();

    private:
        Settings() = default;
        struct State;
        State& state() const;
        void load() const;   // on first use; caller holds the mutex
    };

    inline Settings& settings() { return Settings::instance(); }

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_SETTINGS_HPP
