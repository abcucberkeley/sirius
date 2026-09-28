#include "imgui/settings.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>

#include "imgui/platform.hpp"

namespace sirius::app::gui {

    namespace {

        // What this process set (a value) or removed (nullopt), by key.
        using Changes = std::map<std::string, std::optional<nlohmann::json>>;

        // What tells this process that another one wrote the file: its time
        // and size, from a stat (the file is not read).
        struct Stamp {
            bool exists = false;
            std::filesystem::file_time_type time{};
            std::uintmax_t size = 0;
            bool operator==(const Stamp& o) const { return exists == o.exists && time == o.time && size == o.size; }
        };

        Stamp stampOf(const std::string& path) {
            Stamp s;
            try {
                const std::filesystem::path p = std::filesystem::u8path(path);
                std::error_code ec;
                s.time = std::filesystem::last_write_time(p, ec);
                if (!ec) s.size = std::filesystem::file_size(p, ec);
                s.exists = !ec;
            } catch (const std::exception&) {
                // not UTF-8 (u8path throws on Windows): no file to look at
            }
            return s;
        }

        // Two values the file cannot tell apart: equal, or different only in
        // bytes that are not UTF-8, which it holds as U+FFFD.
        bool sameInFile(const nlohmann::json& a, const nlohmann::json& b) {
            if (a == b) return true;
            constexpr auto replace = nlohmann::json::error_handler_t::replace;
            return a.dump(-1, ' ', false, replace) == b.dump(-1, ' ', false, replace);
        }

        // The settings in the file with this process's unwritten changes on
        // top. Where the file holds this process's own value as writing it
        // came out, the value in memory stays: a Linux path that is not UTF-8
        // would no longer name its file with U+FFFD in it.
        nlohmann::json merged(nlohmann::json file, const nlohmann::json& mine, const Changes& pending) {
            for (auto it = file.begin(); it != file.end(); ++it) {
                const auto own = mine.find(it.key());
                if (own != mine.end() && *own != *it && sameInFile(*own, *it)) *it = *own;
            }
            for (const auto& [key, change] : pending) {
                if (change) file[key] = *change;
                else file.erase(key);
            }
            return file;
        }

    } // namespace

    struct Settings::State {
        std::mutex mutex;
        // Held for the whole of a save, write included: the frame's save and
        // a secret store's save on another thread must not write the file
        // out of order, nor report a write still in flight as done.
        std::mutex saveMutex;
        std::string dir;
        nlohmann::json data = nlohmann::json::object();
        // What this process changed since the file was last written. A save
        // applies it to the file as it is on disk then, so what another
        // instance wrote meanwhile is kept.
        Changes pending;
        bool loaded = false;
        bool dirty = false;
        // A save is between its snapshot and the end of its write: the file
        // may still be the one before it, so reads do not load it meanwhile.
        bool saving = false;
        // The file as this process last read it (nullopt: not known, read it
        // at the next look), and when reads look at it again.
        std::optional<Stamp> seen;
        std::chrono::steady_clock::time_point checkAfter{};
        // After a failed write, autosave() tries the same content again at
        // most once a second (it is called every frame); a new change is
        // tried at once.
        std::chrono::steady_clock::time_point retryAfter{};
        // The failure is said on stderr once, until a write succeeds again.
        // Guarded by saveMutex.
        bool failing = false;
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
        st.pending.clear();
        st.seen.reset();
        st.checkAfter = {};
        st.retryAfter = {};
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
        const auto now = std::chrono::steady_clock::now();
        // Another instance may have written the file since this one read it:
        // a token it stored is then used here too, and a remove() reaches a
        // key only it set. A look is a stat, once a second at most.
        if (st.loaded && (st.saving || now < st.checkAfter)) return;
        st.checkAfter = now + std::chrono::seconds(1);
        const std::string dir = st.dir.empty() ? defaultDirectory() : st.dir;
        const std::string path = dir + "/sirius-app.json";
        const Stamp stamp = stampOf(path);
        if (st.loaded && st.seen && *st.seen == stamp) return;
        st.seen = stamp;
        std::string text;
        if (!st.loaded) {
            st.loaded = true;
            // sirius-imgui.json: the name the file had before the application
            // took over sirius-app's name; read once, then saved under the new one
            if (!platform::readFile(path, text) && !platform::readFile(dir + "/sirius-imgui.json", text)) return;
            const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
            if (j.is_object()) st.data = j;
            return;
        }
        // A file that is gone or does not parse leaves the settings as they are.
        if (!platform::readFile(path, text)) return;
        nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
        if (j.is_object()) st.data = merged(std::move(j), st.data, st.pending);
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
        st.pending[key] = value;
        st.dirty = true;
        st.retryAfter = {};
    }

    void Settings::remove(const std::string& key) {
        State& st = state();
        const std::lock_guard<std::mutex> g(st.mutex);
        // The file as it is now: the key may be one another instance set a
        // moment ago.
        st.checkAfter = {};
        load();
        if (st.data.erase(key) == 0) return;
        st.pending[key] = std::nullopt;
        st.dirty = true;
        st.retryAfter = {};
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

    bool Settings::save() {
        const std::lock_guard<std::mutex> saving(state().saveMutex);
        return saveLocked();
    }

    void Settings::autosave() {
        State& st = state();
        const std::lock_guard<std::mutex> saving(st.saveMutex);
        {
            const std::lock_guard<std::mutex> g(st.mutex);
            if (std::chrono::steady_clock::now() < st.retryAfter) return;
        }
        saveLocked();
    }

    bool Settings::commit(const std::string& key, const std::optional<nlohmann::json>& value) {
        State& st = state();
        // Held from the change to its undo: no other save can write the
        // change meanwhile and leave it in the file after all.
        const std::lock_guard<std::mutex> saving(st.saveMutex);
        std::optional<nlohmann::json> before;
        std::optional<std::optional<nlohmann::json>> pendingBefore;
        {
            const std::lock_guard<std::mutex> g(st.mutex);
            load();
            if (const auto it = st.data.find(key); it != st.data.end()) before = *it;
            if (const auto it = st.pending.find(key); it != st.pending.end()) pendingBefore.emplace(it->second);
            if (value) st.data[key] = *value;
            else st.data.erase(key);
            st.pending[key] = value;
            st.dirty = true;
            st.retryAfter = {};
        }
        if (saveLocked()) return true;
        // Taken back, unless a set() or remove() of the key made meanwhile
        // replaced it. Only memory changes: the value the key had here may be
        // older than the file's, which the next look at the file brings in.
        const std::lock_guard<std::mutex> g(st.mutex);
        const auto it = st.pending.find(key);
        if (it == st.pending.end() || it->second != value) return false;
        if (pendingBefore) it->second = *pendingBefore;
        else st.pending.erase(it);
        if (before) st.data[key] = *before;
        else st.data.erase(key);
        st.dirty = !st.pending.empty();
        return false;
    }

    bool Settings::saveLocked() {
        State& st = state();
        std::string text, dir, path;
        Changes written;
        {
            const std::lock_guard<std::mutex> g(st.mutex);
            if (!st.dirty) return true;
            dir = st.dir.empty() ? defaultDirectory() : st.dir;
            path = dir + "/sirius-app.json";
            // The file as it is now, which another instance may have written
            // since this one read it, with this process's own changes on top.
            // Writing back the copy read at start-up instead erased whatever
            // the other instance had stored, its tokens included. This process
            // sees the other one's keys from here on too.
            nlohmann::json file;
            std::string onDisk;
            if (platform::readFile(path, onDisk)) file = nlohmann::json::parse(onDisk, nullptr, false);
            if (!file.is_object()) file = st.data;
            nlohmann::json all = merged(std::move(file), st.data, st.pending);
            // replace: a string that is not UTF-8 (a Linux file name among the
            // recent datasets) is written with U+FFFD where the strict
            // default threw out of the frame and ended the application.
            text = all.dump(2, ' ', false, nlohmann::json::error_handler_t::replace);
            st.data = std::move(all);
            written.swap(st.pending);
            st.dirty = false;
            st.saving = true;
        }
        const bool ok = platform::makePath(dir) && platform::writeFileAtomic(path, text + "\n");
        const std::lock_guard<std::mutex> g(st.mutex);
        st.saving = false;
        // Written or not, the next look reads the file: another instance may
        // have written it meanwhile.
        st.seen.reset();
        if (ok) {
            st.failing = false;
            return true;
        }
        // Not written: the changes go back for the next try, except where a
        // set() or remove() made meanwhile replaced them.
        for (auto& [key, change] : written) st.pending.try_emplace(key, std::move(change));
        st.dirty = true;
        st.retryAfter = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        if (!st.failing) std::fprintf(stderr, "settings: could not write %s\n", path.c_str());
        st.failing = true;
        return false;
    }

} // namespace sirius::app::gui
