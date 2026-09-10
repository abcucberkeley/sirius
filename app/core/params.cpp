#include "core/params.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace sirius::app {

    using json = nlohmann::json;

    // --- spec builders --------------------------------------------------------

    namespace {
        ParamSpec spec(std::string key, std::string label, ParamType type, ParamValue def) {
            ParamSpec s;
            s.key = std::move(key);
            s.label = std::move(label);
            s.type = type;
            s.defaultValue = std::move(def);
            return s;
        }
    } // namespace

    ParamSpec boolParam(std::string key, std::string label, bool def) {
        return spec(std::move(key), std::move(label), ParamType::Bool, def);
    }
    ParamSpec intParam(std::string key, std::string label, std::int64_t def) {
        return spec(std::move(key), std::move(label), ParamType::Int, def);
    }
    ParamSpec doubleParam(std::string key, std::string label, double def) {
        return spec(std::move(key), std::move(label), ParamType::Double, def);
    }
    ParamSpec stringParam(std::string key, std::string label, std::string def) {
        return spec(std::move(key), std::move(label), ParamType::String, std::move(def));
    }
    ParamSpec pathParam(std::string key, std::string label, std::string def) {
        return spec(std::move(key), std::move(label), ParamType::Path, std::move(def));
    }
    ParamSpec choiceParam(std::string key, std::string label, std::vector<std::string> choices, std::string def) {
        ParamSpec s = spec(std::move(key), std::move(label), ParamType::Choice, std::move(def));
        s.choices = std::move(choices);
        return s;
    }
    ParamSpec channelParam(std::string key, std::string label, std::int64_t def) {
        return spec(std::move(key), std::move(label), ParamType::Channel, def);
    }
    ParamSpec axesParam(std::string key, std::string label, std::string def) {
        return spec(std::move(key), std::move(label), ParamType::Axes, std::move(def));
    }
    ParamSpec doubleListParam(std::string key, std::string label, std::vector<double> def) {
        return spec(std::move(key), std::move(label), ParamType::DoubleList, std::move(def));
    }

    // --- ParamSet -------------------------------------------------------------

    ParamSet::ParamSet(const std::vector<ParamSpec>& specs) {
        for (const ParamSpec& s : specs) items_.emplace_back(s.key, s.defaultValue);
    }

    bool ParamSet::has(const std::string& key) const noexcept { return find(key) != nullptr; }

    const ParamValue* ParamSet::find(const std::string& key) const noexcept {
        for (const auto& kv : items_)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }

    void ParamSet::set(const std::string& key, ParamValue value) {
        for (auto& kv : items_)
            if (kv.first == key) {
                kv.second = std::move(value);
                return;
            }
        items_.emplace_back(key, std::move(value));
    }

    void ParamSet::erase(const std::string& key) {
        items_.erase(std::remove_if(items_.begin(), items_.end(), [&](const auto& kv) { return kv.first == key; }),
                     items_.end());
    }

    bool ParamSet::getBool(const std::string& key, bool def) const {
        const ParamValue* v = find(key);
        if (!v) return def;
        if (const bool* b = std::get_if<bool>(v)) return *b;
        if (const auto* i = std::get_if<std::int64_t>(v)) return *i != 0;
        if (const double* d = std::get_if<double>(v)) return *d != 0.0;
        if (const std::string* s = std::get_if<std::string>(v))
            return *s == "true" || *s == "on" || *s == "1" || *s == "yes";
        return def;
    }

    std::int64_t ParamSet::getInt(const std::string& key, std::int64_t def) const {
        const ParamValue* v = find(key);
        if (!v) return def;
        if (const auto* i = std::get_if<std::int64_t>(v)) return *i;
        if (const double* d = std::get_if<double>(v)) return static_cast<std::int64_t>(std::llround(*d));
        if (const bool* b = std::get_if<bool>(v)) return *b ? 1 : 0;
        if (const std::string* s = std::get_if<std::string>(v)) {
            try {
                return std::stoll(*s);
            } catch (...) { return def; }
        }
        return def;
    }

    double ParamSet::getDouble(const std::string& key, double def) const {
        const ParamValue* v = find(key);
        if (!v) return def;
        if (const double* d = std::get_if<double>(v)) return *d;
        if (const auto* i = std::get_if<std::int64_t>(v)) return static_cast<double>(*i);
        if (const bool* b = std::get_if<bool>(v)) return *b ? 1.0 : 0.0;
        if (const std::string* s = std::get_if<std::string>(v)) {
            try {
                return std::stod(*s);
            } catch (...) { return def; }
        }
        return def;
    }

    std::string ParamSet::getString(const std::string& key, std::string def) const {
        const ParamValue* v = find(key);
        if (!v) return def;
        if (const std::string* s = std::get_if<std::string>(v)) return *s;
        return toDisplayString(*v);
    }

    std::vector<double> ParamSet::getDoubleList(const std::string& key) const {
        const ParamValue* v = find(key);
        if (!v) return {};
        if (const auto* l = std::get_if<std::vector<double>>(v)) return *l;
        if (const double* d = std::get_if<double>(v)) return {*d};
        if (const auto* i = std::get_if<std::int64_t>(v)) return {static_cast<double>(*i)};
        if (const auto* sl = std::get_if<std::vector<std::string>>(v)) {
            std::vector<double> out;
            for (const std::string& s : *sl) {
                try {
                    out.push_back(std::stod(s));
                } catch (...) {}
            }
            return out;
        }
        if (const std::string* s = std::get_if<std::string>(v)) {
            std::vector<double> out;
            std::string tok;
            std::istringstream in(*s);
            while (std::getline(in, tok, ',')) {
                try {
                    out.push_back(std::stod(tok));
                } catch (...) {}
            }
            return out;
        }
        return {};
    }

    std::vector<std::string> ParamSet::getStringList(const std::string& key) const {
        const ParamValue* v = find(key);
        if (!v) return {};
        if (const auto* l = std::get_if<std::vector<std::string>>(v)) return *l;
        if (const std::string* s = std::get_if<std::string>(v)) {
            std::vector<std::string> out;
            std::string tok;
            std::istringstream in(*s);
            while (std::getline(in, tok, ',')) {
                const auto a = tok.find_first_not_of(" \t"), b = tok.find_last_not_of(" \t");
                if (a != std::string::npos) out.push_back(tok.substr(a, b - a + 1));
            }
            return out;
        }
        return {};
    }

    void ParamSet::applyDefaults(const std::vector<ParamSpec>& specs, bool strict) {
        std::vector<std::pair<std::string, ParamValue>> ordered;
        ordered.reserve(specs.size() + items_.size());
        for (const ParamSpec& s : specs) {
            const ParamValue* v = find(s.key);
            ordered.emplace_back(s.key, v ? *v : s.defaultValue);
        }
        if (!strict)
            for (const auto& kv : items_)
                if (std::none_of(specs.begin(), specs.end(), [&](const ParamSpec& s) { return s.key == kv.first; }))
                    ordered.push_back(kv);
        items_ = std::move(ordered);
    }

    void ParamSet::coerce(const std::vector<ParamSpec>& specs) {
        for (const ParamSpec& s : specs) {
            const ParamValue* v = find(s.key);
            if (!v) continue;
            try {
                set(s.key, coerceToSpec(s, sirius::app::toJson(*v)));
            } catch (const std::exception&) {
                set(s.key, s.defaultValue);
            }
        }
    }

    json ParamSet::toJson() const {
        json j = json::object();
        for (const auto& kv : items_) j[kv.first] = sirius::app::toJson(kv.second);
        return j;
    }

    ParamSet ParamSet::fromJson(const json& j) {
        ParamSet p;
        if (!j.is_object()) return p;
        for (auto it = j.begin(); it != j.end(); ++it) p.items_.emplace_back(it.key(), paramValueFromJson(it.value()));
        return p;
    }

    // --- values -----------------------------------------------------------------

    json toJson(const ParamValue& v) {
        return std::visit([](const auto& x) -> json { return json(x); }, v);
    }

    ParamValue paramValueFromJson(const json& j) {
        if (j.is_boolean()) return j.get<bool>();
        if (j.is_number_integer()) return j.get<std::int64_t>();
        if (j.is_number_float()) return j.get<double>();
        if (j.is_string()) return j.get<std::string>();
        if (j.is_array()) {
            if (j.empty()) return std::vector<double>{};
            if (std::all_of(j.begin(), j.end(), [](const json& e) { return e.is_number(); }))
                return j.get<std::vector<double>>();
            std::vector<std::string> out;
            for (const json& e : j) out.push_back(e.is_string() ? e.get<std::string>() : e.dump());
            return out;
        }
        if (j.is_null()) return std::string();
        return j.dump();
    }

    bool ParamSpec::visibleFor(const ParamSet& p) const {
        for (const Visibility& rule : visibility) {
            const ParamValue* v = p.find(rule.key);
            // a rule about a parameter that is not there decides nothing: show
            // the field rather than hide it on a technicality
            if (v == nullptr) continue;
            const std::string current = toDisplayString(*v);
            const bool matches = std::find(rule.values.begin(), rule.values.end(), current) != rule.values.end();
            if (matches == rule.negate) return false;
        }
        return true;
    }

    std::string toDisplayString(const ParamValue& v) {
        struct Visitor {
            std::string operator()(bool b) const { return b ? "on" : "off"; }
            std::string operator()(std::int64_t i) const { return std::to_string(i); }
            std::string operator()(double d) const {
                char buf[32];
                if (d == std::floor(d) && std::abs(d) < 1e9) std::snprintf(buf, sizeof buf, "%.0f", d);
                else std::snprintf(buf, sizeof buf, "%.6g", d);
                return buf;
            }
            std::string operator()(const std::string& s) const { return s; }
            std::string operator()(const std::vector<double>& l) const {
                std::string out;
                for (std::size_t i = 0; i < l.size(); ++i) {
                    if (i) out += ", ";
                    out += (*this)(l[i]);
                }
                return out;
            }
            std::string operator()(const std::vector<std::string>& l) const {
                std::string out;
                for (std::size_t i = 0; i < l.size(); ++i) {
                    if (i) out += ", ";
                    out += l[i];
                }
                return out;
            }
        };
        return std::visit(Visitor{}, v);
    }

    ParamValue coerceToSpec(const ParamSpec& spec, const json& j) {
        auto bad = [&](const char* what) {
            return std::invalid_argument("parameter '" + spec.key + "': expected " + what + ", got " + j.dump());
        };
        auto clampD = [&](double d) { return std::clamp(d, spec.min, spec.max); };
        switch (spec.type) {
            case ParamType::Bool:
                if (j.is_boolean()) return j.get<bool>();
                if (j.is_number()) return j.get<double>() != 0.0;
                if (j.is_string()) {
                    const std::string s = j.get<std::string>();
                    if (s == "true" || s == "on" || s == "1" || s == "yes") return true;
                    if (s == "false" || s == "off" || s == "0" || s == "no") return false;
                }
                throw bad("a boolean");
            case ParamType::Int:
            case ParamType::Channel: {
                double d;
                if (j.is_number()) d = j.get<double>();
                else if (j.is_string()) {
                    try {
                        d = std::stod(j.get<std::string>());
                    } catch (...) { throw bad("an integer"); }
                } else throw bad("an integer");
                if (!std::isfinite(d)) throw bad("an integer");   // llround(NaN) is anything
                return static_cast<std::int64_t>(std::llround(clampD(d)));
            }
            case ParamType::Double: {
                double d;
                if (j.is_number()) d = j.get<double>();
                else if (j.is_string()) {
                    try {
                        d = std::stod(j.get<std::string>());
                    } catch (...) { throw bad("a number"); }
                } else throw bad("a number");
                if (!std::isfinite(d)) throw bad("a finite number");
                return clampD(d);
            }
            case ParamType::String:
            case ParamType::Path:
                if (j.is_string()) return j.get<std::string>();
                if (j.is_null()) return std::string();
                return j.dump();
            case ParamType::Choice: {
                std::string s = j.is_string() ? j.get<std::string>() : j.dump();
                for (const std::string& c : spec.choices)
                    if (c == s) return s;
                // case-insensitive match
                std::string ls = s;
                std::transform(ls.begin(), ls.end(), ls.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                // Case is forgiven; nothing else is. A prefix ("c" for
                // "cubic") used to be, which made a typo a valid setting.
                for (const std::string& c : spec.choices) {
                    std::string lc = c;
                    std::transform(lc.begin(), lc.end(), lc.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                    if (lc == ls) return c;
                }
                std::string opts;
                for (const std::string& c : spec.choices) opts += (opts.empty() ? "" : ", ") + c;
                throw std::invalid_argument("parameter '" + spec.key + "': '" + s + "' is not one of " + opts);
            }
            case ParamType::Axes: {
                std::string s = j.is_string() ? j.get<std::string>() : j.dump();
                std::string out;
                for (char ch : s) {
                    const char l = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                    if (std::string("ctzyx").find(l) != std::string::npos && out.find(l) == std::string::npos) out += l;
                }
                return out;
            }
            case ParamType::DoubleList: {
                if (j.is_array()) {
                    std::vector<double> out;
                    for (const json& e : j) {
                        if (e.is_number()) out.push_back(e.get<double>());
                        else if (e.is_string()) out.push_back(std::stod(e.get<std::string>()));
                        else throw bad("a list of numbers");
                    }
                    return out;
                }
                if (j.is_number()) return std::vector<double>{j.get<double>()};
                if (j.is_string()) {
                    ParamSet tmp;
                    tmp.set("v", j.get<std::string>());
                    return tmp.getDoubleList("v");
                }
                throw bad("a list of numbers");
            }
            case ParamType::StringList: {
                if (j.is_array()) {
                    std::vector<std::string> out;
                    for (const json& e : j) out.push_back(e.is_string() ? e.get<std::string>() : e.dump());
                    return out;
                }
                if (j.is_string()) {
                    ParamSet tmp;
                    tmp.set("v", j.get<std::string>());
                    return tmp.getStringList("v");
                }
                throw bad("a list of strings");
            }
        }
        throw bad("a value");
    }

    json schemaOf(const ParamSpec& spec) {
        json s;
        std::string desc = spec.label;
        if (!spec.unit.empty()) desc += " [" + spec.unit + "]";
        if (!spec.help.empty()) desc += ". " + spec.help;
        switch (spec.type) {
            case ParamType::Bool: s["type"] = "boolean"; break;
            case ParamType::Int:
            case ParamType::Channel:
                s["type"] = "integer";
                if (std::isfinite(spec.min)) s["minimum"] = static_cast<std::int64_t>(spec.min);
                if (std::isfinite(spec.max)) s["maximum"] = static_cast<std::int64_t>(spec.max);
                if (spec.type == ParamType::Channel) desc += " (channel index, 0-based)";
                break;
            case ParamType::Double:
                s["type"] = "number";
                if (std::isfinite(spec.min)) s["minimum"] = spec.min;
                if (std::isfinite(spec.max)) s["maximum"] = spec.max;
                break;
            case ParamType::String:
            case ParamType::Path:
            case ParamType::Axes: s["type"] = "string"; break;
            case ParamType::Choice:
                s["type"] = "string";
                s["enum"] = spec.choices;
                break;
            case ParamType::DoubleList:
                s["type"] = "array";
                s["items"] = {{"type", "number"}};
                break;
            case ParamType::StringList:
                s["type"] = "array";
                s["items"] = {{"type", "string"}};
                break;
        }
        s["description"] = desc;
        return s;
    }

} // namespace sirius::app
