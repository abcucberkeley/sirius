// Multi-file datasets: the manifest that turns a folder of TIFF stacks into
// one (c, t, z, y, x) dataset with tiles, and the filename rule that writes
// it. std::regex has no named groups, so a pattern written with
// (?P<name>...) or (?<name>...) is rewritten to plain groups first
// (plainPattern) and the names are kept by group number.
#include "core/manifest.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#ifndef _WIN32
#include <dirent.h>
#endif

#include <nlohmann/json.hpp>
#include <toml++/toml.hpp>

#include <sirius/tiff_io.hpp>

namespace sirius::app {

    using json = nlohmann::json;
    namespace fs = std::filesystem;

    namespace {

        // --- json <-> toml (the bridge pipeline.cpp uses for pipeline files) ---

        toml::table jsonToToml(const json& j);

        toml::array jsonArrayToToml(const json& j) {
            toml::array a;
            for (const json& e : j) {
                if (e.is_boolean()) a.push_back(e.get<bool>());
                else if (e.is_number_integer()) a.push_back(e.get<std::int64_t>());
                else if (e.is_number_float()) a.push_back(e.get<double>());
                else if (e.is_string()) a.push_back(e.get<std::string>());
                else if (e.is_array()) a.push_back(jsonArrayToToml(e));
                else if (e.is_object()) a.push_back(jsonToToml(e));
                else a.push_back(e.dump());
            }
            return a;
        }

        toml::table jsonToToml(const json& j) {
            toml::table t;
            for (auto it = j.begin(); it != j.end(); ++it) {
                const json& v = it.value();
                if (v.is_boolean()) t.insert(it.key(), v.get<bool>());
                else if (v.is_number_integer()) t.insert(it.key(), v.get<std::int64_t>());
                else if (v.is_number_float()) t.insert(it.key(), v.get<double>());
                else if (v.is_string()) t.insert(it.key(), v.get<std::string>());
                else if (v.is_array()) t.insert(it.key(), jsonArrayToToml(v));
                else if (v.is_object()) t.insert(it.key(), jsonToToml(v));
                else t.insert(it.key(), v.dump());
            }
            return t;
        }

        json tomlToJson(const toml::node& n) {
            if (const auto* v = n.as_boolean()) return v->get();
            if (const auto* v = n.as_integer()) return v->get();
            if (const auto* v = n.as_floating_point()) return v->get();
            if (const auto* v = n.as_string()) return v->get();
            if (const auto* a = n.as_array()) {
                json out = json::array();
                for (const toml::node& e : *a) out.push_back(tomlToJson(e));
                return out;
            }
            if (const auto* t = n.as_table()) {
                json out = json::object();
                for (const auto& [k, v] : *t) out[std::string(k.str())] = tomlToJson(v);
                return out;
            }
            if (const auto* d = n.as_date()) {
                std::ostringstream ss;
                ss << *d;
                return ss.str();
            }
            if (const auto* t = n.as_time()) {
                std::ostringstream ss;
                ss << *t;
                return ss.str();
            }
            if (const auto* dt = n.as_date_time()) {
                std::ostringstream ss;
                ss << *dt;
                return ss.str();
            }
            return json();
        }

        // --- small helpers ---

        bool isDigit(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }

        bool isInteger(const std::string& s) {
            std::size_t i = (!s.empty() && (s[0] == '-' || s[0] == '+')) ? 1 : 0;
            if (i >= s.size()) return false;
            for (; i < s.size(); ++i)
                if (!isDigit(s[i])) return false;
            return true;
        }

        bool isNumber(const std::string& s) {
            if (s.empty()) return false;
            char* end = nullptr;
            std::strtod(s.c_str(), &end);
            return end && *end == '\0';
        }

        // "tile_x2_y2" before "tile_x10_y2": runs of digits compare by value,
        // so channel "2" sorts before "10" and grid names keep grid order.
        bool naturalLess(const std::string& a, const std::string& b) {
            std::size_t i = 0, j = 0;
            while (i < a.size() && j < b.size()) {
                if (isDigit(a[i]) && isDigit(b[j])) {
                    std::size_t i2 = i, j2 = j;
                    while (i2 < a.size() && isDigit(a[i2])) ++i2;
                    while (j2 < b.size() && isDigit(b[j2])) ++j2;
                    std::string na = a.substr(i, i2 - i), nb = b.substr(j, j2 - j);
                    na.erase(0, std::min(na.find_first_not_of('0'), na.size() - 1));
                    nb.erase(0, std::min(nb.find_first_not_of('0'), nb.size() - 1));
                    if (na.size() != nb.size()) return na.size() < nb.size();
                    if (na != nb) return na < nb;
                    i = i2;
                    j = j2;
                    continue;
                }
                if (a[i] != b[j]) return a[i] < b[j];
                ++i;
                ++j;
            }
            return (a.size() - i) < (b.size() - j);
        }

        std::string folderName(const fs::path& folder) {
            fs::path p = folder;
            if (p.filename().empty()) p = p.parent_path();   // "/data/exp/" -> "exp"
            return p.filename().string();
        }

        // Case-insensitive ".tif" / ".tiff".
        bool isTiffName(const fs::path& p) {
            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return ext == ".tif" || ext == ".tiff";
        }

        std::string sizeText(std::uint32_t w, std::uint32_t h, std::size_t pages) {
            return std::to_string(w) + " x " + std::to_string(h) + " x " + std::to_string(pages) + " pages";
        }

        // Hand-written manifests: a string field may arrive as a number
        // ("label = 405", "exposure = 8") and a number as a string; anything
        // else is reported by field name.
        std::string stringField(const json& j, const char* key, const std::string& fallback = {}) {
            if (!j.contains(key) || j[key].is_null()) return fallback;
            const json& v = j[key];
            if (v.is_string()) return v.get<std::string>();
            if (v.is_number_integer()) return std::to_string(v.get<long long>());
            if (v.is_number_float()) {
                std::ostringstream ss;
                ss << v.get<double>();
                return ss.str();
            }
            throw std::invalid_argument(std::string("'") + key + "' must be a string, not " + v.type_name());
        }

        double numberField(const json& j, const char* key, double fallback) {
            if (!j.contains(key) || j[key].is_null()) return fallback;
            const json& v = j[key];
            if (v.is_number()) return v.get<double>();
            if (v.is_string()) {
                try {
                    return std::stod(v.get<std::string>());
                } catch (const std::exception&) {
                }
            }
            throw std::invalid_argument(std::string("'") + key + "' must be a number, not " + v.type_name());
        }

        json channelToJson(const ChannelInfo& c) {
            return json{{"label", c.label}, {"wavelength_nm", c.wavelengthNm}, {"color", c.hexColor()}, {"exposure", c.exposure}};
        }

        ChannelInfo channelFromJson(const json& j) {
            ChannelInfo c;
            c.label = stringField(j, "label");
            c.wavelengthNm = numberField(j, "wavelength_nm", 0.0);
            c.exposure = stringField(j, "exposure");
            const std::string hex = stringField(j, "color");
            if (!hex.empty()) c.color = colorFromHex(hex);
            else if (c.wavelengthNm > 0.0) c.color = colorForWavelength(c.wavelengthNm);
            return c;
        }

        json tileToJson(const TileInfo& t) {
            return json{{"name", t.name},
                        {"position_um", json::array({t.positionUm[0], t.positionUm[1], t.positionUm[2]})},
                        {"grid", json::array({t.gridIndex[0], t.gridIndex[1], t.gridIndex[2]})}};
        }

        TileInfo tileFromJson(const json& j) {
            TileInfo t;
            t.name = stringField(j, "name");
            if (j.contains("position_um") && j["position_um"].is_array() && j["position_um"].size() == 3)
                for (std::size_t k = 0; k < 3; ++k) t.positionUm[k] = j["position_um"][k].get<double>();
            if (j.contains("grid") && j["grid"].is_array() && j["grid"].size() == 3)
                for (std::size_t k = 0; k < 3; ++k) t.gridIndex[k] = j["grid"][k].get<Index>();
            return t;
        }

    } // namespace

    // --- DatasetManifest --------------------------------------------------------------

    fs::path manifestFilePath(const fs::path& folder, const ManifestFile& file) {
        fs::path p(file.path);
        return p.is_absolute() ? p : folder / p;
    }

    fs::path DatasetManifest::filesRoot(const fs::path& loadedFrom) const {
        std::error_code ec;
        const fs::path home = fs::is_directory(loadedFrom, ec) ? loadedFrom : loadedFrom.parent_path();
        if (filesFolder.empty()) return home;
        // A relative files_folder is relative to the manifest, as the paths of
        // a pipeline file are to it; it used to resolve against the working
        // directory, so the same sidecar worked or not by where the application
        // was started.
        const fs::path root(filesFolder);
        return root.is_absolute() ? root : (home / root).lexically_normal();
    }

    Index DatasetManifest::channelIndex(const std::string& channel) const noexcept {
        for (std::size_t i = 0; i < channels.size(); ++i)
            if (channels[i].label == channel) return static_cast<Index>(i);
        // an index written as text, for manifests without channel names
        if (isInteger(channel)) {
            const long long k = std::strtoll(channel.c_str(), nullptr, 10);
            if (k >= 0 && k < static_cast<long long>(channels.size())) return static_cast<Index>(k);
        }
        return -1;
    }

    Index DatasetManifest::tileIndex(const std::string& tile) const noexcept {
        if (tile.empty()) return tiles.empty() ? -1 : 0;
        for (std::size_t i = 0; i < tiles.size(); ++i)
            if (tiles[i].name == tile) return static_cast<Index>(i);
        return -1;
    }

    Index DatasetManifest::timePoints() const noexcept {
        Index n = 0;
        for (const ManifestFile& f : files) n = std::max(n, f.t + 1);
        return n;
    }

    const ManifestFile* DatasetManifest::file(Index tile, Index channel, Index t) const noexcept {
        for (const ManifestFile& f : files)
            if (f.t == t && tileIndex(f.tile) == tile && channelIndex(f.channel) == channel) return &f;
        return nullptr;
    }

    json DatasetManifest::toJson() const {
        json j;
        j["name"] = name;
        j["voxel_um"] = json::array({voxelUm[0], voxelUm[1], voxelUm[2]});
        j["frame_interval_s"] = frameIntervalS;
        j["acquisition"] = acquisition;
        j["pattern"] = pattern;
        if (!filesFolder.empty()) j["files_folder"] = filesFolder;
        j["sim"] = json{{"present", sim.present}, {"ndirs", sim.ndirs}, {"nphases", sim.nphases}, {"fast_si", sim.fastSi}};
        j["channels"] = json::array();
        for (const ChannelInfo& c : channels) j["channels"].push_back(channelToJson(c));
        j["tiles"] = json::array();
        for (const TileInfo& t : tiles) j["tiles"].push_back(tileToJson(t));
        j["files"] = json::array();
        for (const ManifestFile& f : files)
            j["files"].push_back(json{{"path", f.path}, {"channel", f.channel}, {"t", f.t}, {"tile", f.tile}});
        return j;
    }

    DatasetManifest DatasetManifest::fromJson(const json& j) {
        DatasetManifest m;
        m.name = stringField(j, "name");
        if (j.contains("voxel_um") && j["voxel_um"].is_array() && j["voxel_um"].size() == 3)
            for (std::size_t k = 0; k < 3; ++k) m.voxelUm[k] = j["voxel_um"][k].get<double>();
        m.frameIntervalS = numberField(j, "frame_interval_s", 0.0);
        m.acquisition = stringField(j, "acquisition");
        m.pattern = stringField(j, "pattern");
        m.filesFolder = stringField(j, "files_folder");
        if (j.contains("sim") && j["sim"].is_object()) {
            const json& s = j["sim"];
            m.sim.present = s.value("present", false);
            m.sim.ndirs = s.value("ndirs", 3);
            m.sim.nphases = s.value("nphases", 5);
            m.sim.fastSi = s.value("fast_si", false);
        }
        if (j.contains("channels"))
            for (const json& c : j["channels"]) m.channels.push_back(channelFromJson(c));
        if (j.contains("tiles"))
            for (const json& t : j["tiles"]) m.tiles.push_back(tileFromJson(t));
        if (j.contains("files"))
            for (const json& f : j["files"]) {
                ManifestFile mf;
                mf.path = stringField(f, "path");
                mf.channel = stringField(f, "channel");
                mf.t = static_cast<Index>(numberField(f, "t", 0.0));
                mf.tile = stringField(f, "tile");
                m.files.push_back(std::move(mf));
            }
        return m;
    }

    void DatasetManifest::save(const fs::path& path) const {
        std::error_code ec;
        const fs::path target = fs::is_directory(path, ec) ? path / kFileName : path;
        toml::table t = jsonToToml(toJson());
        t.insert("format", "sirius-dataset");
        t.insert("version", std::int64_t{1});
        std::ofstream out(target);
        if (!out) throw std::runtime_error("cannot write " + target.string());
        out << "# SIRIUS multi-file dataset: one TIFF stack per channel, time point and tile.\n"
            << t << "\n";
        if (!out) throw std::runtime_error("cannot write " + target.string());
    }

    DatasetManifest DatasetManifest::load(const fs::path& path) {
        std::error_code ec;
        const fs::path source = fs::is_directory(path, ec) ? path / kFileName : path;
        toml::table t;
        try {
            t = toml::parse_file(source.string());
        } catch (const toml::parse_error& e) {
            std::ostringstream ss;
            ss << source.string() << ": " << e.description() << " (line " << e.source().begin.line << ")";
            throw std::runtime_error(ss.str());
        }
        try {
            return fromJson(tomlToJson(t));
        } catch (const std::exception& e) {
            throw std::runtime_error(source.string() + ": " + e.what());
        }
    }

    std::vector<std::string> DatasetManifest::validate(const fs::path& folder) const {
        std::vector<std::string> problems;
        // a broken folder can produce one line per file; keep the report readable
        constexpr std::size_t kMaxProblems = 50;
        std::size_t hidden = 0;
        auto add = [&](std::string s) {
            if (problems.size() < kMaxProblems) problems.push_back(std::move(s));
            else ++hidden;
        };
        if (channels.empty()) add("The manifest lists no channels.");
        if (tiles.empty()) add("The manifest lists no tiles.");
        if (files.empty()) add("The manifest lists no files.");
        std::set<std::string> seen;
        for (const ChannelInfo& c : channels)
            if (!seen.insert(c.label).second) add("Duplicate channel '" + c.label + "'.");
        seen.clear();
        for (const TileInfo& t : tiles)
            if (!seen.insert(t.name).second) add("Duplicate tile '" + t.name + "'.");
        std::map<std::tuple<Index, Index, Index>, std::string> slots;
        for (const ManifestFile& f : files) {
            std::error_code ec;
            if (!fs::is_regular_file(manifestFilePath(folder, f), ec)) add("Missing file: " + f.path);
            const Index c = channelIndex(f.channel), k = tileIndex(f.tile);
            if (c < 0) add("Unknown channel '" + f.channel + "' for " + f.path);
            if (k < 0) add("Unknown tile '" + f.tile + "' for " + f.path);
            if (f.t < 0) add("Negative time point for " + f.path);
            if (c < 0 || k < 0 || f.t < 0) continue;
            auto [it, fresh] = slots.emplace(std::make_tuple(k, c, f.t), f.path);
            if (!fresh)
                add("Both " + it->second + " and " + f.path + " map to tile '" + tiles[static_cast<std::size_t>(k)].name +
                    "', channel '" + channels[static_cast<std::size_t>(c)].label + "', t " + std::to_string(f.t) + ".");
        }
        const Index nt = timePoints();
        for (std::size_t k = 0; k < tiles.size(); ++k)
            for (std::size_t c = 0; c < channels.size(); ++c)
                for (Index t = 0; t < nt; ++t)
                    if (!slots.count(std::make_tuple(static_cast<Index>(k), static_cast<Index>(c), t)))
                        add("No file for tile '" + tiles[k].name + "', channel '" + channels[c].label + "', t " +
                            std::to_string(t) + ".");
        if (hidden > 0) problems.push_back("... and " + std::to_string(hidden) + " more.");
        return problems;
    }

    // --- filename patterns ------------------------------------------------------------

    std::string plainPattern(const std::string& pattern, std::vector<std::string>* groupNames) {
        std::string out;
        out.reserve(pattern.size());
        std::vector<std::string> names;
        bool inClass = false;
        const std::size_t n = pattern.size();
        for (std::size_t i = 0; i < n; ++i) {
            const char ch = pattern[i];
            if (ch == '\\') {
                // an escape: copy it and the escaped character, which may be a
                // parenthesis or bracket that must not open a group / class
                out += ch;
                if (i + 1 < n) out += pattern[++i];
                continue;
            }
            if (inClass) {
                if (ch == ']') inClass = false;
                out += ch;
                continue;
            }
            if (ch == '[') {
                inClass = true;
                out += ch;
                continue;
            }
            if (ch != '(') {
                out += ch;
                continue;
            }
            if (i + 1 < n && pattern[i + 1] == '?') {
                std::size_t j = i + 2;
                if (j < n && pattern[j] == 'P') ++j;   // Python's (?P<name>...)
                const bool named = j + 1 < n && pattern[j] == '<' && pattern[j + 1] != '=' && pattern[j + 1] != '!';
                if (named) {
                    const std::size_t end = pattern.find('>', j + 1);
                    if (end == std::string::npos) throw std::invalid_argument("unterminated group name in pattern: " + pattern);
                    const std::string name = pattern.substr(j + 1, end - j - 1);
                    const bool wellFormed = !name.empty() && !isDigit(name[0]) &&
                                            std::all_of(name.begin(), name.end(), [](char c) {
                                                return isDigit(c) || std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
                                            });
                    if (!wellFormed) throw std::invalid_argument("bad group name '" + name + "' in pattern: " + pattern);
                    names.push_back(name);
                    out += '(';
                    i = end;
                    continue;
                }
                // (?:...), lookahead, ...: no capture, copied as written
                out += ch;
                continue;
            }
            names.emplace_back();   // an unnamed group still takes a group number
            out += ch;
        }
        if (groupNames) *groupNames = std::move(names);
        return out;
    }

    std::vector<FilenameMatch> matchFilenames(const std::vector<std::string>& names, const std::string& pattern) {
        std::vector<std::string> groups;
        const std::string plain = plainPattern(pattern, &groups);
        std::regex re;
        try {
            re = std::regex(plain, std::regex::ECMAScript);
        } catch (const std::regex_error& e) {
            throw std::invalid_argument("pattern does not compile: " + std::string(e.what()));
        }
        std::vector<FilenameMatch> out;
        out.reserve(names.size());
        for (const std::string& name : names) {
            FilenameMatch m;
            m.file = name;
            std::smatch sm;
            // search, not match: the pattern need not spell out the whole
            // name (anchor with ^ and $ when it should)
            if (std::regex_search(name, sm, re)) {
                m.matched = true;
                for (std::size_t k = 0; k < groups.size() && k + 1 < sm.size(); ++k)
                    if (!groups[k].empty() && sm[k + 1].matched) m.groups[groups[k]] = sm[k + 1].str();
            }
            out.push_back(std::move(m));
        }
        return out;
    }

    // --- manifest from a folder -------------------------------------------------------

    namespace {

        struct MatchedFile {
            std::string name;
            std::string channelToken;
            Index t = 0;
            std::string tile;
            std::string x, y, z;   // grid / stage tokens, empty when absent
        };

        std::string tokenOf(const std::map<std::string, std::string>& g, const char* a, const char* b = nullptr) {
            auto it = g.find(a);
            if (it != g.end()) return it->second;
            if (b) {
                it = g.find(b);
                if (it != g.end()) return it->second;
            }
            return {};
        }

        Index parseIndex(const std::string& token, const char* what, const std::string& file) {
            if (!isInteger(token))
                throw std::invalid_argument(file + ": " + what + " '" + token + "' is not an integer");
            return static_cast<Index>(std::strtoll(token.c_str(), nullptr, 10));
        }

        double parseMicrons(const std::string& token, const char* what, const std::string& file) {
            if (!isNumber(token))
                throw std::invalid_argument(file + ": " + what + " '" + token + "' is not a number");
            return std::strtod(token.c_str(), nullptr);
        }

        // Rank of every distinct coordinate along one axis: stage positions
        // become a grid for the tile map even though they are not indices.
        std::map<double, Index> ranks(const std::vector<double>& values) {
            std::set<double> distinct(values.begin(), values.end());
            std::map<double, Index> out;
            Index i = 0;
            for (double v : distinct) out[v] = i++;
            return out;
        }

    } // namespace

    // naturalLess above is what puts "f2" before "f10".
    // Names only: QDir and directory_entry::is_regular_file stat every entry.
    // On NFS / Vast, readdir reports DT_UNKNOWN, so a Files filter skips the
    // TIFFs unless each one is stated — and stating hundreds of 300 MB stacks
    // hangs the GUI. Extension of the name is enough here.
    std::vector<std::string> tiffNamesInOrder(const fs::path& folder) {
        std::vector<std::string> names;
#ifdef _WIN32
        std::error_code ec;
        for (const fs::directory_entry& e : fs::directory_iterator(folder, ec)) {
            if (!e.is_regular_file(ec) || !isTiffName(e.path())) continue;
            const std::string name = e.path().filename().string();
            if (!name.empty() && name.front() == '.') continue;
            names.push_back(name);
        }
#else
        DIR* dir = ::opendir(folder.string().c_str());
        if (!dir) return names;
        while (const dirent* ent = ::readdir(dir)) {
            const char* n = ent->d_name;
            if (n[0] == '.') continue;   // ".", ".." and resource forks
            if (!isTiffName(fs::path(n))) continue;
            // Names only, no stat per file: that is what made a folder of ten
            // thousand files on NFS take minutes. The entry's own type is free
            // where the filesystem gives one, and rules out a directory named
            // like a TIFF; a link or an unknown type costs one stat.
#ifdef DT_DIR
            if (ent->d_type == DT_DIR) continue;
            if (ent->d_type == DT_LNK || ent->d_type == DT_UNKNOWN) {
                std::error_code ec;
                if (!fs::is_regular_file(folder / n, ec)) continue;   // a dangling link, a link to a folder
            }
#endif
            names.emplace_back(n);
        }
        ::closedir(dir);
#endif
        std::sort(names.begin(), names.end(), naturalLess);
        return names;
    }

    DatasetManifest manifestOfOneStack(const fs::path& folder) {
        DatasetManifest m;
        m.name = folderName(folder);
        m.acquisition = "One stack per file";
        m.channels.push_back(ChannelInfo{});
        m.tiles.push_back(TileInfo{});
        Index t = 0;
        for (const std::string& name : tiffNamesInOrder(folder)) {
            ManifestFile f;
            f.path = name;
            f.channel = m.channels.front().label;
            f.t = t++;
            m.files.push_back(std::move(f));
        }
        return m;
    }

    DatasetManifest manifestFromFolder(const fs::path& folder, const FilenameRule& rule, std::vector<std::string>* unmatched) {
        std::error_code ec;
        if (!fs::is_directory(folder, ec)) throw std::runtime_error("not a folder: " + folder.string());
        std::vector<std::string> names = tiffNamesInOrder(folder);

        if (unmatched) unmatched->clear();
        std::vector<MatchedFile> matched;
        for (const FilenameMatch& m : matchFilenames(names, rule.pattern)) {
            if (!m.matched) {
                if (unmatched) unmatched->push_back(m.file);
                continue;
            }
            MatchedFile f;
            f.name = m.file;
            f.channelToken = tokenOf(m.groups, "channel", "c");
            if (f.channelToken.empty()) f.channelToken = "0";
            const std::string t = tokenOf(m.groups, "t", "time");
            f.t = t.empty() ? 0 : parseIndex(t, "time point", m.file);
            f.x = tokenOf(m.groups, "x", "col");
            f.y = tokenOf(m.groups, "y", "row");
            f.z = tokenOf(m.groups, "z");
            f.tile = tokenOf(m.groups, "tile");
            if (f.tile.empty()) {
                // no tile group: a name from the grid tokens, or the single tile
                f.tile = "tile";
                if (!f.x.empty()) f.tile += "_x" + f.x;
                if (!f.y.empty()) f.tile += "_y" + f.y;
                if (!f.z.empty()) f.tile += "_z" + f.z;
            }
            matched.push_back(std::move(f));
        }

        DatasetManifest manifest;
        manifest.name = folderName(folder);
        manifest.voxelUm = rule.voxelUm;
        manifest.frameIntervalS = rule.frameIntervalS;
        manifest.sim = rule.sim;
        manifest.acquisition = rule.acquisition;
        manifest.pattern = rule.pattern;
        if (matched.empty()) return manifest;

        // channels, in natural token order; names from the rule when given
        std::vector<std::string> tokens;
        for (const MatchedFile& f : matched) tokens.push_back(f.channelToken);
        std::sort(tokens.begin(), tokens.end(), naturalLess);
        tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
        std::map<std::string, std::string> labelOfToken;
        for (const std::string& token : tokens) {
            ChannelInfo ch;
            auto it = rule.channelInfo.find(token);
            if (it != rule.channelInfo.end()) {
                ch = it->second;
                if (ch.label.empty()) ch.label = token;
            } else {
                ch.label = token;
                if (isNumber(token)) {
                    const double nm = std::strtod(token.c_str(), nullptr);
                    if (nm >= 300.0 && nm <= 900.0) ch.wavelengthNm = nm;   // "488" is an emission line
                }
            }
            if (ch.wavelengthNm > 0.0 && ch.color == std::array<float, 3>{1.f, 1.f, 1.f})
                ch.color = colorForWavelength(ch.wavelengthNm);
            labelOfToken[token] = ch.label;
            manifest.channels.push_back(std::move(ch));
        }

        // tiles, in natural name order, with the first file of each
        std::map<std::string, const MatchedFile*> firstOfTile;
        std::vector<std::string> tileNames;
        for (const MatchedFile& f : matched)
            if (firstOfTile.emplace(f.tile, &f).second) tileNames.push_back(f.tile);
        std::sort(tileNames.begin(), tileNames.end(), naturalLess);

        // Shape of the dataset: one (z, y, x) for every tile, channel and time
        // point. Probe the first file of each tile — a diSPIM / AOLLS folder is
        // hundreds of 800-page stacks, and inspectTiff of every file walks every
        // IFD of every file (and used to get the process killed).
        std::uint32_t width = 0, height = 0;
        std::size_t pages = 0;
        std::string firstFile;
        auto probeShape = [&](const MatchedFile& f) {
            const TiffStackShape info = inspectTiffShape((folder / f.name).string());
            if (info.pages == 0) throw std::runtime_error(f.name + ": the TIFF has no pages");
            if (firstFile.empty()) {
                width = info.width;
                height = info.height;
                pages = info.pages;
                firstFile = f.name;
                return;
            }
            if (info.width != width || info.height != height || info.pages != pages)
                throw std::runtime_error("tile shape mismatch: " + f.name + " is " + sizeText(info.width, info.height, info.pages) +
                                         ", " + firstFile + " is " + sizeText(width, height, pages));
        };
        if (!matched.empty()) probeShape(matched.front());
        for (const std::string& name : tileNames) probeShape(*firstOfTile[name]);

        std::vector<double> xs, ys, zs;   // Microns: for the grid ranks
        if (rule.positions == FilenameRule::Positions::Microns) {
            for (const std::string& name : tileNames) {
                const MatchedFile& f = *firstOfTile[name];
                xs.push_back(f.x.empty() ? 0.0 : parseMicrons(f.x, "x position", f.name));
                ys.push_back(f.y.empty() ? 0.0 : parseMicrons(f.y, "y position", f.name));
                zs.push_back(f.z.empty() ? 0.0 : parseMicrons(f.z, "z position", f.name));
            }
        }
        const std::map<double, Index> xRank = ranks(xs), yRank = ranks(ys), zRank = ranks(zs);
        const std::array<double, 3> extentUm{static_cast<double>(pages) * rule.voxelUm[2], static_cast<double>(height) * rule.voxelUm[1],
                                             static_cast<double>(width) * rule.voxelUm[0]};   // z, y, x
        const double step = 1.0 - rule.overlapFraction;
        for (std::size_t i = 0; i < tileNames.size(); ++i) {
            const MatchedFile& f = *firstOfTile[tileNames[i]];
            TileInfo tile;
            tile.name = tileNames[i];
            switch (rule.positions) {
                case FilenameRule::Positions::GridIndex: {
                    const bool anyIndex = !f.x.empty() || !f.y.empty() || !f.z.empty();
                // a `tile` group without grid tokens: lay the tiles out in a row
                    tile.gridIndex = {f.z.empty() ? 0 : parseIndex(f.z, "z index", f.name),
                                      f.y.empty() ? 0 : parseIndex(f.y, "y index", f.name),
                                      f.x.empty() ? (anyIndex ? 0 : static_cast<Index>(i)) : parseIndex(f.x, "x index", f.name)};
                    for (std::size_t k = 0; k < 3; ++k)
                        tile.positionUm[k] = static_cast<double>(tile.gridIndex[k]) * extentUm[k] * step;
                    break;
                }
                case FilenameRule::Positions::Microns:
                    tile.positionUm = {zs[i], ys[i], xs[i]};
                    tile.gridIndex = {zRank.at(zs[i]), yRank.at(ys[i]), xRank.at(xs[i])};
                    break;
                case FilenameRule::Positions::None:
                    tile.gridIndex = {0, 0, static_cast<Index>(i)};
                    break;
            }
            manifest.tiles.push_back(std::move(tile));
        }

        std::map<std::tuple<std::string, std::string, Index>, std::string> slots;
        for (const MatchedFile& f : matched) {
            ManifestFile mf;
            mf.path = f.name;
            mf.channel = labelOfToken[f.channelToken];
            mf.t = f.t;
            mf.tile = f.tile;
            auto [it, fresh] = slots.emplace(std::make_tuple(mf.tile, mf.channel, mf.t), f.name);
            if (!fresh)
                throw std::runtime_error("both " + it->second + " and " + f.name + " map to tile '" + mf.tile + "', channel '" +
                                         mf.channel + "', t " + std::to_string(mf.t) + "; the pattern needs another group");
            manifest.files.push_back(std::move(mf));
        }
        return manifest;
    }

} // namespace sirius::app
