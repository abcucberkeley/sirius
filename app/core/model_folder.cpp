#include "core/model_folder.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <mutex>
#include <system_error>
#include <tuple>

#include "core/host.hpp"
#include "core/params.hpp"

namespace sirius::app {

    namespace {

        namespace fs = std::filesystem;
        using nlohmann::json;

        constexpr const char* kFormat = "latents-model/";
        constexpr int kFormatMajor = 1;

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        std::string trim(const std::string& s) {
            const std::size_t a = s.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) return {};
            return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
        }

        std::string text(const json& j, const char* key) {
            const auto it = j.find(key);
            return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
        }

        std::vector<double> numbers(const json& j, const char* key) {
            std::vector<double> out;
            const auto it = j.find(key);
            if (it == j.end() || !it->is_array()) return out;
            for (const json& v : *it)
                if (v.is_number()) out.push_back(v.get<double>());
            return out;
        }

        // README.md's first paragraph that is not a heading, as one line.
        std::string readmeLine(const fs::path& folder) {
            std::string all;
            if (!host::readFile((folder / "README.md").u8string(), all)) return {};
            std::string out, line;
            std::size_t at = 0;
            bool fenced = false;
            while (at <= all.size()) {
                const std::size_t end = std::min(all.find('\n', at), all.size());
                line = trim(all.substr(at, end - at));
                at = end + 1;
                if (line.rfind("```", 0) == 0) fenced = !fenced;
                const bool skip = fenced || line.empty() || line[0] == '#' || line.rfind("```", 0) == 0;
                if (skip) {
                    if (!out.empty()) break;
                    continue;
                }
                out += (out.empty() ? "" : " ") + line;
            }
            return out;
        }

        // The facts of a parsed model.json; nullopt with the reason when it is not one.
        std::optional<ModelFolderFacts> fromManifest(const json& man, const std::string& where, std::string* error) {
            const auto fail = [&](const std::string& why) -> std::optional<ModelFolderFacts> {
                if (error) *error = why;
                return std::nullopt;
            };
            if (!man.is_object()) return fail(where + " does not describe a model.");
            const std::string format = text(man, "format");
            if (format.rfind(kFormat, 0) != 0)
                return fail(where + " is format '" + (format.empty() ? std::string("?") : format) +
                            "', not latents-model/1: not a model the Foundation step runs.");
            int major = -1;
            try {
                major = std::stoi(format.substr(std::string(kFormat).size()));
            } catch (...) {
                major = -1;
            }
            if (major != kFormatMajor)
                return fail(where + " is " + format + "; this SIRIUS reads latents-model/1. Update SIRIUS, or export the model with the matching latents.");
            ModelFolderFacts f;
            f.name = text(man, "name");
            f.version = text(man, "version");
            if (const auto t = man.find("tasks"); t != man.end() && t->is_array())
                for (const json& v : *t)
                    if (v.is_string()) f.tasks.push_back(v.get<std::string>());
            if (f.tasks.empty()) return fail(where + " lists no tasks.");
            f.promptable = f.offers("prompt");
            f.description = trim(text(man, "notes"));
            if (const auto in = man.find("input"); in != man.end() && in->is_object()) {
                std::vector<double> zyx = numbers(*in, "voxel_um");
                if (zyx.size() == 3) f.voxelUm = {zyx[2], zyx[1], zyx[0]};
                if (const auto c = in->find("channels"); c != in->end() && c->is_number_integer()) f.channels = std::max(1, c->get<int>());
                f.channelMerge = text(*in, "channel_merge");
            }
            return f;
        }

    } // namespace

    bool ModelFolderFacts::offers(const std::string& task) const { return std::find(tasks.begin(), tasks.end(), task) != tasks.end(); }

    std::string ModelFolderFacts::title() const {
        std::string out = name.empty() ? std::string("model") : name;
        if (!version.empty()) out += " " + version;
        std::string t;
        for (const std::string& task : tasks) t += (t.empty() ? "" : ", ") + task;
        if (!t.empty()) out += " \xC2\xB7 " + t;
        return out;
    }

    std::string modelTaskOfLabel(const std::string& label) { return label == kPromptTask ? "prompt" : "segment"; }

    std::string modelTaskLabel(const std::string& task) {
        if (task == "segment") return kModelSegmentLabel;
        if (task == "prompt") return kPromptTask;
        return {};
    }

    std::vector<std::string> modelTaskChoices(const std::optional<ModelFolderFacts>& facts) {
        std::vector<std::string> out;
        for (const char* task : {"segment", "prompt"})
            if (!facts || facts->offers(task)) out.push_back(modelTaskLabel(task));
        if (out.empty()) out.push_back(kModelSegmentLabel);   // a model of tasks the step has none of: its error says so
        return out;
    }

    bool isOldBundlePath(const std::string& path) {
        const std::string l = lower(path);
        return l.size() >= 4 && l.compare(l.size() - 4, 4, ".ltb") == 0;
    }

    std::string oldBundleMessage(const std::string& path) {
        return "Old bundle format: " + fs::u8path(path).filename().u8string() +
               " is a .ltb bundle, which needed the latents package. Models are self-contained folders now: re-export it with "
               "latents scripts/export_model.py (<models>/<name>/<version>/ with model.py, model.json and weights.safetensors) and "
               "choose that folder.";
    }

    std::optional<ModelFolderFacts> readModelFolder(const std::string& path, std::string* error) {
        const auto fail = [&](const std::string& why) -> std::optional<ModelFolderFacts> {
            if (error) *error = why;
            return std::nullopt;
        };
        if (trim(path).empty()) return fail("Choose a model folder (one holding model.py and model.json).");
        if (isOldBundlePath(path)) return fail(oldBundleMessage(path));
        std::error_code ec;
        fs::path folder = fs::u8path(path);
        if (lower(folder.filename().u8string()) == "model.json" && fs::is_regular_file(folder, ec)) folder = folder.parent_path();
        if (!fs::exists(folder, ec)) return fail("Model folder not found: " + path + ".");
        if (!fs::is_directory(folder, ec))
            return fail(path + " is a file, not a model folder: choose the folder that holds model.py and model.json.");
        const fs::path manifest = folder / "model.json";
        if (!fs::is_regular_file(manifest, ec))
            return fail(path + " is not a model folder: it has no model.json. A model folder holds model.py, model.json and "
                               "weights.safetensors (latents scripts/export_model.py writes one); Models\xE2\x80\xA6 lists the models of a folder.");

        // cached by the file's stamp: the parameter panel asks every frame
        static std::mutex mutex;
        static std::map<std::string, std::tuple<fs::file_time_type, std::uintmax_t, std::optional<ModelFolderFacts>, std::string>> cache;
        const fs::file_time_type stamp = fs::last_write_time(manifest, ec);
        const std::uintmax_t size = fs::file_size(manifest, ec);
        const std::string key = manifest.u8string();
        {
            const std::lock_guard<std::mutex> g(mutex);
            if (const auto it = cache.find(key); it != cache.end() && std::get<0>(it->second) == stamp && std::get<1>(it->second) == size) {
                if (!std::get<2>(it->second)) return fail(std::get<3>(it->second));
                ModelFolderFacts f = *std::get<2>(it->second);
                f.path = path;
                return f;
            }
        }
        std::string body, why;
        std::optional<ModelFolderFacts> facts;
        if (!host::readFile(key, body)) {
            why = "Cannot read " + key + ".";
        } else {
            const json man = json::parse(body, nullptr, false);
            if (man.is_discarded()) why = key + " is not valid JSON: the export did not finish, or the file was edited.";
            else facts = fromManifest(man, key, &why);
        }
        if (facts) {
            if (facts->name.empty()) facts->name = folder.parent_path().filename().u8string();
            if (facts->version.empty()) facts->version = folder.filename().u8string();
            if (facts->description.empty()) facts->description = readmeLine(folder);
        }
        {
            const std::lock_guard<std::mutex> g(mutex);
            if (cache.size() > 64) cache.clear();
            cache[key] = {stamp, size, facts, why};
        }
        if (!facts) return fail(why);
        facts->path = path;
        return facts;
    }

    ModelListing listModelFolders(const std::vector<std::string>& dirs) {
        ModelListing out;
        std::vector<std::string> seen;
        std::error_code ec;
        const auto add = [&](const fs::path& folder) {
            const std::string key = folder.lexically_normal().u8string();
            if (std::find(seen.begin(), seen.end(), key) != seen.end()) return;
            seen.push_back(key);
            std::string why;
            std::optional<ModelFolderFacts> f = readModelFolder(folder.u8string(), &why);
            ModelFolderFacts entry;
            if (f) {
                entry = std::move(*f);
            } else {
                entry.path = folder.u8string();
                entry.name = folder.parent_path().filename().u8string();
                entry.version = folder.filename().u8string();
                entry.error = why;
            }
            std::error_code e2;
            const std::uintmax_t size = fs::file_size(folder / "weights.safetensors", e2);
            entry.sizeBytes = e2 ? -1 : static_cast<long long>(size);
            out.models.push_back(std::move(entry));
        };
        // the visible subfolders, by name
        const auto subdirs = [&](const fs::path& d) {
            std::vector<fs::path> got;
            std::error_code e2;
            for (fs::directory_iterator it(d, e2), end; !e2 && it != end; it.increment(e2)) {
                std::error_code e3;
                const std::string n = it->path().filename().u8string();
                if (!n.empty() && n[0] != '.' && n[0] != '_' && it->is_directory(e3)) got.push_back(it->path());
            }
            std::sort(got.begin(), got.end(), [](const fs::path& a, const fs::path& b) { return lower(a.filename().u8string()) < lower(b.filename().u8string()); });
            return got;
        };
        for (const std::string& raw : dirs) {
            const std::string d = trim(raw);
            if (d.empty()) continue;
            const fs::path dir = fs::u8path(d);
            if (!fs::is_directory(dir, ec)) {
                out.errors.push_back("not a folder: " + d);
                continue;
            }
            if (fs::is_regular_file(dir / "model.json", ec)) {
                add(dir);
                continue;
            }
            std::vector<fs::path> old;
            for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
                std::error_code e3;
                if (it->is_regular_file(e3) && isOldBundlePath(it->path().filename().u8string())) old.push_back(it->path());
            }
            std::sort(old.begin(), old.end());
            for (const fs::path& o : old) {
                ModelFolderFacts entry;
                entry.path = o.u8string();
                entry.name = o.stem().u8string();
                entry.error = oldBundleMessage(entry.path);
                out.models.push_back(std::move(entry));
            }
            for (const fs::path& name : subdirs(dir)) {
                if (fs::is_regular_file(name / "model.json", ec)) {
                    add(name);
                    continue;
                }
                for (const fs::path& version : subdirs(name))
                    if (fs::is_regular_file(version / "model.json", ec)) add(version);
            }
        }
        return out;
    }

    ModelListing modelListingFromJson(const json& reply) {
        ModelListing out;
        if (const auto e = reply.find("errors"); e != reply.end() && e->is_array())
            for (const json& v : *e)
                if (v.is_string()) out.errors.push_back(v.get<std::string>());
        const auto list = reply.find("models");
        if (list == reply.end() || !list->is_array()) return out;
        for (const json& m : *list) {
            if (!m.is_object()) continue;
            std::string why;
            std::optional<ModelFolderFacts> f = modelFactsFromJson(m, &why);
            ModelFolderFacts entry;
            if (f) {
                entry = std::move(*f);
            } else {
                entry.path = text(m, "path");
                entry.name = text(m, "name");
                entry.version = text(m, "version");
                entry.error = why;
            }
            if (const auto s = m.find("size_bytes"); s != m.end() && s->is_number_integer()) entry.sizeBytes = s->get<long long>();
            out.models.push_back(std::move(entry));
        }
        return out;
    }

    std::optional<ModelFolderFacts> modelFactsFromJson(const json& info, std::string* error) {
        const auto fail = [&](const std::string& why) -> std::optional<ModelFolderFacts> {
            if (error) *error = why;
            return std::nullopt;
        };
        if (!info.is_object()) return fail("the worker's answer describes no model");
        if (const std::string e = text(info, "error"); !e.empty()) return fail(e);
        if (text(info, "format") != "latents-model") return fail("not a model folder (format '" + text(info, "format") + "')");
        ModelFolderFacts f;
        f.path = text(info, "path");
        f.name = text(info, "name");
        f.version = text(info, "version");
        f.description = text(info, "description");
        if (const auto t = info.find("tasks"); t != info.end() && t->is_array())
            for (const json& v : *t)
                if (v.is_string()) f.tasks.push_back(v.get<std::string>());
        if (f.tasks.empty()) return fail("the model lists no tasks");
        f.promptable = f.offers("prompt");
        f.voxelUm = numbers(info, "voxel_um");
        if (f.voxelUm.size() != 3) f.voxelUm.clear();
        if (const auto c = info.find("channels"); c != info.end() && c->is_number_integer()) f.channels = std::max(1, c->get<int>());
        f.channelMerge = text(info, "channel_merge");
        return f;
    }

} // namespace sirius::app
