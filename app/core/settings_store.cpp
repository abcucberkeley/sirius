#include "core/settings_store.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>

#include "core/host.hpp"
#include "core/settings_toml.hpp"

namespace sirius::app {

    namespace platform = host;

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
        // sirius-app.json (or sirius-imgui.json), read at start-up because
        // sirius-app.toml was not there: renamed to <name>.migrated once its
        // settings are written as TOML.
        std::string legacyFile;
        // The file as last read did not parse: what toml++ said, and where.
        // It is not written over while this is set.
        std::string loadError;
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
        constexpr const char* kFileName = "sirius-app.toml";

        std::string defaultDirectory() {
            const std::string base = platform::configDirectory();
            return base.empty() ? std::string("sirius") : base + "/sirius";
        }

        std::string errorText(const settings_toml::ParseResult& r) {
            if (r.line <= 0) return r.error;
            return "line " + std::to_string(r.line) + ", column " + std::to_string(r.column) + ": " + r.error;
        }

        // The secrets/ entries of a JSON file of before (Windows' DPAPI
        // blobs, base64) into secrets.json beside it, whose entries the
        // secret store reads by the same names; never into the TOML file.
        // An entry already there stays.
        bool moveSecretsOut(const std::string& dir, const nlohmann::json& old) {
            nlohmann::json secrets = nlohmann::json::object();
            for (auto it = old.begin(); it != old.end(); ++it)
                if (it.key().rfind("secrets/", 0) == 0 && it.value().is_string()) secrets[it.key().substr(8)] = it.value();
            if (secrets.empty()) return true;
            const std::string path = dir + "/secrets.json";
            std::string text;
            nlohmann::json store = nlohmann::json::object();
            if (platform::readFile(path, text) && !text.empty()) {
                store = nlohmann::json::parse(text, nullptr, false);
                if (!store.is_object()) return false;   // never written over
            }
            for (auto it = secrets.begin(); it != secrets.end(); ++it)
                if (!store.contains(it.key())) store[it.key()] = it.value();
            return platform::writeFileAtomic(path, store.dump(4) + "\n", true);
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
        st.legacyFile.clear();
        st.loadError.clear();
    }

    std::string Settings::directory() const {
        State& st = state();
        const std::lock_guard<std::mutex> g(st.mutex);
        return st.dir.empty() ? defaultDirectory() : st.dir;
    }

    std::string Settings::filePath() const { return directory() + "/" + kFileName; }

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
        const std::string path = dir + "/" + kFileName;
        const Stamp stamp = stampOf(path);
        if (st.loaded && st.seen && *st.seen == stamp) return;
        st.seen = stamp;
        std::string text;
        if (!st.loaded) {
            st.loaded = true;
            // The settings hold tokens and the paths of plugins and models: a
            // folder others may write is one they could put their own in.
            std::string why;
            if (host::writableByOthers(dir, &why))
                std::fprintf(stderr, "settings: WARNING: %s is not private (%s); run chmod 700 on it\n", dir.c_str(), why.c_str());
            if (platform::readFile(path, text)) {
                const settings_toml::ParseResult r = settings_toml::fromToml(text);
                if (r.ok) {
                    st.data = r.flat;
                } else {
                    // the defaults, and the file left as it is for the user to fix
                    st.loadError = errorText(r);
                    std::fprintf(stderr, "settings: %s does not read (%s); starting with the defaults, the file is left as it is\n",
                                 path.c_str(), st.loadError.c_str());
                }
                return;
            }
            if (stamp.exists) return;   // there, but not readable: nothing is taken from elsewhere
            // The JSON file of before (sirius-app.json, or sirius-imgui.json
            // before the application took sirius-app's name). Its settings
            // become this process's changes, so the first save writes them as
            // TOML, and the old file is renamed to <name>.migrated then.
            for (const char* name : {"sirius-app.json", "sirius-imgui.json"}) {
                const std::string legacy = dir + "/" + name;
                if (!platform::readFile(legacy, text)) continue;
                const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
                if (!j.is_object()) continue;
                if (!moveSecretsOut(dir, j))
                    std::fprintf(stderr, "settings: could not move the secrets of %s into secrets.json\n", legacy.c_str());
                st.data = j;
                for (auto it = j.begin(); it != j.end(); ++it) st.pending.try_emplace(it.key(), *it);
                st.dirty = true;
                st.legacyFile = legacy;
                return;
            }
            return;
        }
        // A file that is gone leaves the settings as they are; one that does
        // not parse too, and is said.
        if (!platform::readFile(path, text)) return;
        const settings_toml::ParseResult r = settings_toml::fromToml(text);
        if (!r.ok) {
            st.loadError = errorText(r);
            return;
        }
        st.loadError.clear();
        st.data = merged(r.flat, st.data, st.pending);
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
        std::string text, dir, path, legacy;
        Changes written;
        {
            const std::lock_guard<std::mutex> g(st.mutex);
            if (!st.dirty) return true;
            dir = st.dir.empty() ? defaultDirectory() : st.dir;
            path = dir + "/" + kFileName;
            // The file as it is now, which another instance may have written
            // since this one read it, with this process's own changes on top.
            // Writing back the copy read at start-up instead erased whatever
            // the other instance had stored, its tokens included. This process
            // sees the other one's keys from here on too.
            nlohmann::json file = nlohmann::json::object();
            std::string onDisk;
            if (stampOf(path).exists) {
                // There but not readable (a lock, a permission): nothing is
                // written over it now; the next save tries again.
                if (!platform::readFile(path, onDisk)) {
                    st.retryAfter = std::chrono::steady_clock::now() + std::chrono::seconds(1);
                    if (!st.failing) std::fprintf(stderr, "settings: could not read %s; not saved yet\n", path.c_str());
                    st.failing = true;
                    return false;
                }
                const settings_toml::ParseResult r = settings_toml::fromToml(onDisk);
                if (!r.ok) {
                    // A hand edit that does not parse is the user's to fix: never written over.
                    st.loadError = errorText(r);
                    st.retryAfter = std::chrono::steady_clock::now() + std::chrono::seconds(1);
                    if (!st.failing)
                        std::fprintf(stderr, "settings: %s does not read (%s); not saved until it does\n", path.c_str(), st.loadError.c_str());
                    st.failing = true;
                    return false;
                }
                st.loadError.clear();
                file = r.flat;
            }
            // A missing file (removed to reset the settings) starts from
            // nothing: only this process's own changes are written, not the
            // whole copy it read at start-up.
            legacy = st.legacyFile;
            nlohmann::json all = merged(std::move(file), st.data, st.pending);
            // replace: a string that is not UTF-8 (a Linux file name among the
            // recent datasets) is written with U+FFFD where the strict
            // default threw out of the frame and ended the application.
            text = settings_toml::toToml(all);
            st.data = std::move(all);
            written.swap(st.pending);
            st.dirty = false;
            st.saving = true;
        }
        const bool ok = platform::makePath(dir) && platform::writeFileAtomic(path, text);
        const std::lock_guard<std::mutex> g(st.mutex);
        st.saving = false;
        // Written or not, the next look reads the file: another instance may
        // have written it meanwhile.
        st.seen.reset();
        if (ok) {
            st.failing = false;
            if (!legacy.empty() && legacy == st.legacyFile) {
                // kept beside it under another name: a downgrade can take it back
                std::error_code ec;
                std::filesystem::rename(std::filesystem::u8path(legacy), std::filesystem::u8path(legacy + ".migrated"), ec);
                if (ec) std::fprintf(stderr, "settings: could not rename %s to %s.migrated\n", legacy.c_str(), legacy.c_str());
                st.legacyFile.clear();
            }
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

    std::string Settings::loadError() const {
        State& st = state();
        const std::lock_guard<std::mutex> g(st.mutex);
        load();
        return st.loadError;
    }

    bool Settings::adoptText(const std::string& text, std::string* error) {
        State& st = state();
        const settings_toml::ParseResult r = settings_toml::fromToml(text);
        if (!r.ok) {
            if (error) *error = errorText(r);
            return false;
        }
        const std::lock_guard<std::mutex> saving(st.saveMutex);
        std::string dir, path;
        {
            const std::lock_guard<std::mutex> g(st.mutex);
            dir = st.dir.empty() ? defaultDirectory() : st.dir;
            path = dir + "/" + kFileName;
        }
        if (!platform::makePath(dir) || !platform::writeFileAtomic(path, text)) {
            if (error) *error = "could not write " + path;
            return false;
        }
        const std::lock_guard<std::mutex> g(st.mutex);
        // What is not in the file stays: the secrets of a migration, in
        // memory until the secret store moves them. The rest is the text's.
        nlohmann::json data = r.flat;
        for (auto it = st.data.begin(); it != st.data.end(); ++it)
            if (it.key().rfind("secrets/", 0) == 0 && !data.contains(it.key())) data[it.key()] = it.value();
        st.data = std::move(data);
        st.pending.clear();
        st.dirty = false;
        st.loadError.clear();
        st.failing = false;
        st.retryAfter = {};
        st.seen = stampOf(path);
        st.checkAfter = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        return true;
    }

} // namespace sirius::app
