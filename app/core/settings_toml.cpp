#include "core/settings_toml.hpp"

#include <cstdint>
#include <map>
#include <sstream>

#include <toml++/toml.hpp>

namespace sirius::app::settings_toml {

    using json = nlohmann::json;

    namespace {

        constexpr const char* kHeader =
            "# SIRIUS settings.\n"
            "#\n"
            "# SIRIUS reads this file when it starts and writes it when a setting\n"
            "# changes. Edit it with Preferences > Edit settings file..., which checks\n"
            "# what you type before saving and applies it at once. Comments added by\n"
            "# hand are not kept when SIRIUS saves (it writes the file anew, with the\n"
            "# notes below). Secrets -- tokens, passwords, API keys -- are never\n"
            "# written here: SIRIUS keeps them in its secret store.\n";

        // A line or two above each group SIRIUS writes; other groups get none.
        const std::map<std::string, std::string>& groupNotes() {
            static const std::map<std::string, std::string> notes = {
                {"cluster",
                 "# Clusters: one [cluster.<name>] table per cluster profile (Process >\n"
                 "# Connect to cluster...), with the partitions its dropdowns offer as\n"
                 "# [[cluster.<name>.partitions]]. docs/clusters.example.toml explains each key."},
                {"compute", "# The backend and the devices the steps run on."},
                {"worker", "# The local Python worker (segmentation models, user operations)."},
                {"window", "# The window's size and which panels are shown."},
                {"ui", "# The interface's scale and floating windows."},
                {"recent", "# Recently opened datasets."},
                {"assistant", "# The assistant (its API key is in the secret store, not here)."},
                {"hpc", "# A worker you started and tunnelled yourself (its token is in the secret store)."},
                {"hub", "# The model hub (its token is in the secret store)."},
                {"folderDataset", "# Folder datasets: the file name patterns used last."},
            };
            return notes;
        }

        std::string sanitized(const std::string& s) { return validUtf8(s); }

        toml::table toTable(const json& j);

        toml::array toArray(const json& j) {
            toml::array a;
            for (const json& e : j) {
                if (e.is_null()) continue;
                if (e.is_boolean()) a.push_back(e.get<bool>());
                else if (e.is_number_unsigned()) a.push_back(static_cast<std::int64_t>(e.get<std::uint64_t>()));
                else if (e.is_number_integer()) a.push_back(e.get<std::int64_t>());
                else if (e.is_number_float()) a.push_back(e.get<double>());
                else if (e.is_string()) a.push_back(sanitized(e.get<std::string>()));
                else if (e.is_array()) a.push_back(toArray(e));
                else if (e.is_object()) a.push_back(toTable(e));
            }
            return a;
        }

        void insert(toml::table& t, const std::string& key, const json& v) {
            const std::string k = sanitized(key);
            if (v.is_null()) return;
            if (v.is_boolean()) t.insert_or_assign(k, v.get<bool>());
            else if (v.is_number_unsigned()) t.insert_or_assign(k, static_cast<std::int64_t>(v.get<std::uint64_t>()));
            else if (v.is_number_integer()) t.insert_or_assign(k, v.get<std::int64_t>());
            else if (v.is_number_float()) t.insert_or_assign(k, v.get<double>());
            else if (v.is_string()) t.insert_or_assign(k, sanitized(v.get<std::string>()));
            else if (v.is_array()) t.insert_or_assign(k, toArray(v));
            else if (v.is_object()) t.insert_or_assign(k, toTable(v));
        }

        toml::table toTable(const json& j) {
            toml::table t;
            for (auto it = j.begin(); it != j.end(); ++it) insert(t, it.key(), it.value());
            return t;
        }

        json toJson(const toml::node& n) {
            if (const auto* v = n.as_boolean()) return v->get();
            if (const auto* v = n.as_integer()) return v->get();
            if (const auto* v = n.as_floating_point()) return v->get();
            if (const auto* v = n.as_string()) return v->get();
            if (const auto* a = n.as_array()) {
                json out = json::array();
                for (const toml::node& e : *a) out.push_back(toJson(e));
                return out;
            }
            if (const auto* t = n.as_table()) {
                json out = json::object();
                for (const auto& [k, v] : *t) out[std::string(k.str())] = toJson(v);
                return out;
            }
            // dates and times, which SIRIUS never writes: as text
            std::ostringstream ss;
            if (const auto* d = n.as_date()) ss << *d;
            else if (const auto* t = n.as_time()) ss << *t;
            else if (const auto* dt = n.as_date_time()) ss << *dt;
            return ss.str();
        }

        std::string format(const toml::table& t) {
            std::ostringstream out;
            constexpr auto flags = toml::format_flags::allow_literal_strings | toml::format_flags::allow_multi_line_strings |
                                   toml::format_flags::allow_unicode_strings | toml::format_flags::allow_real_tabs_in_strings |
                                   toml::format_flags::indent_array_elements;
            out << toml::toml_formatter{t, flags};
            return out.str();
        }

    } // namespace

    std::string validUtf8(const std::string& text) {
        std::string out;
        out.reserve(text.size());
        const auto* s = reinterpret_cast<const unsigned char*>(text.data());
        const std::size_t n = text.size();
        std::size_t i = 0;
        while (i < n) {
            const unsigned char c = s[i];
            std::size_t len = 0;
            std::uint32_t cp = 0;
            if (c < 0x80) {
                out.push_back(static_cast<char>(c));
                ++i;
                continue;
            }
            if ((c & 0xE0) == 0xC0) {
                len = 2;
                cp = c & 0x1F;
            } else if ((c & 0xF0) == 0xE0) {
                len = 3;
                cp = c & 0x0F;
            } else if ((c & 0xF8) == 0xF0) {
                len = 4;
                cp = c & 0x07;
            }
            bool ok = len > 0 && i + len <= n;
            for (std::size_t k = 1; ok && k < len; ++k) {
                if ((s[i + k] & 0xC0) != 0x80) ok = false;
                else cp = (cp << 6) | (s[i + k] & 0x3F);
            }
            // overlong forms, surrogates and what is past U+10FFFF are not UTF-8 either
            if (ok && ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) || cp > 0x10FFFF ||
                       (cp >= 0xD800 && cp <= 0xDFFF)))
                ok = false;
            if (ok) {
                out.append(text, i, len);
                i += len;
            } else {
                out += "\xEF\xBF\xBD";
                ++i;
            }
        }
        return out;
    }

    std::string toToml(const json& flat, bool header) {
        // top-level keys (no slash) first, then one table per group
        toml::table top;
        std::map<std::string, json> groups;
        if (flat.is_object())
            for (auto it = flat.begin(); it != flat.end(); ++it) {
                const std::string& key = it.key();
                if (key.rfind("secrets/", 0) == 0 || it.value().is_null()) continue;
                const std::size_t slash = key.find('/');
                if (slash == std::string::npos || slash == 0) {
                    insert(top, key, it.value());
                    continue;
                }
                json& g = groups[key.substr(0, slash)];
                if (!g.is_object()) g = json::object();
                g[key.substr(slash + 1)] = it.value();
            }
        std::string out = header ? std::string(kHeader) : std::string();
        if (!top.empty()) out += "\n" + format(top) + "\n";
        for (const auto& [name, values] : groups) {
            toml::table one;
            insert(one, name, values);
            if (one.empty()) continue;
            out += "\n";
            if (const auto note = groupNotes().find(name); note != groupNotes().end()) out += note->second + "\n";
            out += format(one) + "\n";
        }
        return out;
    }

    ParseResult fromToml(const std::string& text) {
        ParseResult r;
        toml::table t;
        try {
            t = toml::parse(text);
        } catch (const toml::parse_error& e) {
            r.error = std::string(e.description());
            r.line = static_cast<int>(e.source().begin.line);
            r.column = static_cast<int>(e.source().begin.column);
            return r;
        }
        for (const auto& [k, v] : t) {
            const std::string key(k.str());
            if (const auto* group = v.as_table()) {
                for (const auto& [k2, v2] : *group) r.flat[key + "/" + std::string(k2.str())] = toJson(v2);
            } else {
                r.flat[key] = toJson(v);
            }
        }
        r.ok = true;
        return r;
    }

    std::optional<std::pair<int, int>> position(const std::string& text, const std::vector<std::string>& path) {
        toml::table t;
        try {
            t = toml::parse(text);
        } catch (const toml::parse_error&) {
            return std::nullopt;
        }
        const toml::node* node = &t;
        std::optional<std::pair<int, int>> best;
        for (const std::string& part : path) {
            const toml::node* next = nullptr;
            if (const auto* tab = node->as_table()) {
                next = tab->get(part);
            } else if (const auto* arr = node->as_array()) {
                std::size_t idx = 0;
                bool number = !part.empty();
                for (const char c : part) number = number && c >= '0' && c <= '9';
                if (number) {
                    idx = static_cast<std::size_t>(std::stoull(part));
                    if (idx < arr->size()) next = arr->get(idx);
                }
            }
            if (!next) break;
            node = next;
            const toml::source_region& src = node->source();
            if (src.begin.line > 0) best = std::make_pair(static_cast<int>(src.begin.line), static_cast<int>(src.begin.column));
        }
        return best;
    }

    std::pair<std::string, std::vector<std::string>> splitPath(const std::vector<std::string>& path) {
        if (path.empty()) return {};
        if (path.size() == 1) return {path[0], {}};
        return {path[0] + "/" + path[1], std::vector<std::string>(path.begin() + 2, path.end())};
    }

} // namespace sirius::app::settings_toml
