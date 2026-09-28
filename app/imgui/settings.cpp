#include "imgui/settings.hpp"

#include <mutex>

#include "imgui/platform.hpp"

namespace sirius::app::gui {

    struct Settings::State {
        std::mutex mutex;
        std::string dir;
        nlohmann::json data = nlohmann::json::object();
        bool loaded = false;
        bool dirty = false;
    };

    Settings& Settings::instance() {
        static Settings s;
        return s;
    }

    Settings::State& Settings::state() const {
        static State st;
        return st;
    }

    namespace {
        std::string defaultDirectory() {
            const std::string base = platform::configDirectory();
            return base.empty() ? std::string("sirius") : base + "/sirius";
        }
    } // namespace

    void Settings::setDirectory(const std::string& dir) {
        State& st = state();
        const std::lock_guard<std::mutex> g(st.mutex);
        st.dir = dir;
        st.loaded = false;
        st.dirty = false;
        st.data = nlohmann::json::object();
    }

    std::string Settings::directory() const {
        State& st = state();
        const std::lock_guard<std::mutex> g(st.mutex);
        return st.dir.empty() ? defaultDirectory() : st.dir;
    }

    std::string Settings::filePath() const { return directory() + "/sirius-app.json"; }

    std::string Settings::layoutPath() const { return directory() + "/imgui.ini"; }

    void Settings::load() const {
        State& st = state();
        if (st.loaded) return;
        st.loaded = true;
        const std::string dir = st.dir.empty() ? defaultDirectory() : st.dir;
        std::string text;
        // sirius-imgui.json: the name the file had before the application took
        // over sirius-app's name; read once, then saved under the new one
        if (!platform::readFile(dir + "/sirius-app.json", text) && !platform::readFile(dir + "/sirius-imgui.json", text)) return;
        const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
        if (j.is_object()) st.data = j;
    }

    bool Settings::contains(const std::string& key) const {
        State& st = state();
        const std::lock_guard<std::mutex> g(st.mutex);
        load();
        return st.data.contains(key);
    }

    nlohmann::json Settings::value(const std::string& key, const nlohmann::json& def) const {
        State& st = state();
        const std::lock_guard<std::mutex> g(st.mutex);
        load();
        const auto it = st.data.find(key);
        return it == st.data.end() ? def : *it;
    }

    std::string Settings::getString(const std::string& key, const std::string& def) const {
        const nlohmann::json v = value(key);
        if (v.is_string()) return v.get<std::string>();
        if (v.is_number() || v.is_boolean()) return v.dump();
        return def;
    }

    int Settings::getInt(const std::string& key, int def) const {
        const nlohmann::json v = value(key);
        if (v.is_number()) return static_cast<int>(v.get<double>());
        if (v.is_boolean()) return v.get<bool>() ? 1 : 0;
        if (v.is_string()) {
            try {
                return std::stoi(v.get<std::string>());
            } catch (const std::exception&) {
                return def;
            }
        }
        return def;
    }

    double Settings::getDouble(const std::string& key, double def) const {
        const nlohmann::json v = value(key);
        if (v.is_number()) return v.get<double>();
        if (v.is_string()) {
            try {
                return std::stod(v.get<std::string>());
            } catch (const std::exception&) {
                return def;
            }
        }
        return def;
    }

    bool Settings::getBool(const std::string& key, bool def) const {
        const nlohmann::json v = value(key);
        if (v.is_boolean()) return v.get<bool>();
        if (v.is_number()) return v.get<double>() != 0.0;
        if (v.is_string()) return v.get<std::string>() == "true" || v.get<std::string>() == "1";
        return def;
    }

    std::vector<std::string> Settings::getStringList(const std::string& key) const {
        std::vector<std::string> out;
        const nlohmann::json v = value(key);
        if (v.is_array()) {
            for (const auto& e : v)
                if (e.is_string()) out.push_back(e.get<std::string>());
        } else if (v.is_string() && !v.get<std::string>().empty()) {
            out.push_back(v.get<std::string>());
        }
        return out;
    }

    void Settings::set(const std::string& key, const nlohmann::json& value) {
        State& st = state();
        const std::lock_guard<std::mutex> g(st.mutex);
        load();
        const auto it = st.data.find(key);
        if (it != st.data.end() && *it == value) return;
        st.data[key] = value;
        st.dirty = true;
    }

    void Settings::remove(const std::string& key) {
        State& st = state();
        const std::lock_guard<std::mutex> g(st.mutex);
        load();
        if (st.data.erase(key) > 0) st.dirty = true;
    }

    std::vector<std::string> Settings::keys(const std::string& prefix) const {
        State& st = state();
        const std::lock_guard<std::mutex> g(st.mutex);
        load();
        std::vector<std::string> out;
        for (auto it = st.data.begin(); it != st.data.end(); ++it)
            if (it.key().compare(0, prefix.size(), prefix) == 0) out.push_back(it.key());
        return out;
    }

    void Settings::save() {
        State& st = state();
        std::string text, dir;
        {
            const std::lock_guard<std::mutex> g(st.mutex);
            if (!st.dirty) return;
            st.dirty = false;
            text = st.data.dump(2);
            dir = st.dir.empty() ? defaultDirectory() : st.dir;
        }
        platform::makePath(dir);
        platform::writeFileAtomic(dir + "/sirius-app.json", text + "\n");
    }

} // namespace sirius::app::gui
