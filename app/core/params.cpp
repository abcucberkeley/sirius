#include "core/params.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
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
    ParamSpec promptsParam(std::string key, std::string label) {
        return spec(std::move(key), std::move(label), ParamType::Prompts, ParamJson{"[]"});
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
        if (const ParamJson* j = std::get_if<ParamJson>(v)) return j->text;   // the JSON, not its summary
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

    namespace {
        struct ToJson {
            template <class T>
            json operator()(const T& x) const {
                return json(x);
            }
            json operator()(const ParamJson& x) const {
                // canonical text, written by coerceToSpec; text that is not
                // JSON (a hand-edited file) stays visible as the string it is
                json out = json::parse(x.text, nullptr, false);
                return out.is_discarded() ? json(x.text) : out;
            }
        };
    } // namespace

    json toJson(const ParamValue& v) { return std::visit(ToJson{}, v); }

    ParamValue paramValueFromJson(const json& j) {
        if (j.is_boolean()) return j.get<bool>();
        if (j.is_number_integer()) return j.get<std::int64_t>();
        if (j.is_number_float()) return j.get<double>();
        if (j.is_string()) return j.get<std::string>();
        if (j.is_array()) {
            if (j.empty()) return std::vector<double>{};
            if (std::all_of(j.begin(), j.end(), [](const json& e) { return e.is_number(); }))
                return j.get<std::vector<double>>();
            // a list of records (the points of a Prompt step) stays structured
            if (std::any_of(j.begin(), j.end(), [](const json& e) { return e.is_object(); })) return ParamJson{j.dump()};
            std::vector<std::string> out;
            for (const json& e : j) out.push_back(e.is_string() ? e.get<std::string>() : e.dump());
            return out;
        }
        if (j.is_null()) return std::string();
        if (j.is_object()) return ParamJson{j.dump()};
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
            // What a person reads in an undo entry or a tool's change list:
            // "2 objects: 1 box, 2 points", not the JSON of them.
            std::string operator()(const ParamJson& v) const {
                const json j = json::parse(v.text, nullptr, false);
                if (!j.is_array()) return v.text;
                if (j.empty()) return "none";
                std::size_t points = 0, boxes = 0, scribbles = 0, other = 0;
                std::vector<std::int64_t> objects;
                for (const json& e : j) {
                    if (e.is_object() && e.contains("object") && e["object"].is_number_integer()) {
                        const std::int64_t id = e["object"].get<std::int64_t>();
                        if (std::find(objects.begin(), objects.end(), id) == objects.end()) objects.push_back(id);
                    }
                    const std::string kind = e.is_object() && e.contains("kind") && e["kind"].is_string() ? e["kind"].get<std::string>()
                                             : e.is_object() && e.contains("x")                           ? std::string("point")
                                                                                                          : std::string();
                    if (kind == "point") ++points;
                    else if (kind == "box") ++boxes;
                    else if (kind == "scribble") ++scribbles;
                    else ++other;
                }
                std::string out;
                const auto part = [&out](std::size_t n, const char* one, const char* many) {
                    if (n == 0) return;
                    out += (out.empty() ? "" : ", ") + std::to_string(n) + " " + (n == 1 ? one : many);
                };
                part(boxes, "box", "boxes");
                part(points, "point", "points");
                part(scribbles, "scribble", "scribbles");
                part(other, "entry", "entries");
                if (!objects.empty()) out = std::to_string(objects.size()) + (objects.size() == 1 ? " object: " : " objects: ") + out;
                return out;
            }
        };
        return std::visit(Visitor{}, v);
    }

    namespace {
        // A coordinate as it is stored: an integer when it is one, so a point
        // clicked on voxel 12 reads 12 in the file and in get_step, not 12.0.
        json storedNumber(double v) {
            if (std::floor(v) == v && std::abs(v) < 1e15) return static_cast<std::int64_t>(v);
            return v;
        }

        // A list of three coordinates, each a voxel position >= 0.
        json storedTriple(const json& e, const std::function<std::invalid_argument(const std::string&)>& bad, const std::string& which) {
            if (!e.is_array() || e.size() != 3) throw bad(which + " is not [x, y, z]: " + e.dump());
            json out = json::array();
            for (const json& v : e) {
                if (!v.is_number() || !std::isfinite(v.get<double>()) || v.get<double>() < 0.0)
                    throw bad(which + " has a coordinate that is not a voxel position >= 0: " + e.dump());
                out.push_back(storedNumber(v.get<double>()));
            }
            return out;
        }

        // A canonical record (coercePrompts) as a Prompt.
        Prompt promptOfRecord(const json& e) {
            const std::string kind = e.at("kind").get<std::string>();
            const std::int64_t t = e.at("t").get<std::int64_t>();
            const std::uint32_t id = e.contains("object") ? e["object"].get<std::uint32_t>() : 0u;
            if (kind == "box")
                return Prompt::boxOf({e.at("x0").get<double>(), e.at("y0").get<double>(), e.at("z0").get<double>(), e.at("x1").get<double>(),
                                      e.at("y1").get<double>(), e.at("z1").get<double>()},
                                     t, id);
            if (kind == "scribble") {
                std::vector<std::array<double, 3>> stroke;
                for (const json& q : e.at("points")) stroke.push_back({q[0].get<double>(), q[1].get<double>(), q[2].get<double>()});
                return Prompt::scribble(std::move(stroke), t, e.at("label").get<int>() != 0, id);
            }
            return Prompt::point(e.at("x").get<double>(), e.at("y").get<double>(), e.at("z").get<double>(), t, e.at("label").get<int>() != 0, id);
        }

        // The object ids a list written before objects did not have
        // (params.hpp): an object prompt starts an object of its own,
        // numbered after the highest id in the list; a background prompt
        // joins the object nearest to it on its time point, or, on a time
        // point without any, the one object that holds that time point's
        // stray background prompts and is never sent.
        void assignPromptObjects(const json& records, std::vector<std::uint32_t>& ids) {
            if (std::none_of(ids.begin(), ids.end(), [](std::uint32_t id) { return id == 0; })) return;
            std::vector<Prompt> prompts;
            for (const json& e : records) prompts.push_back(promptOfRecord(e));
            std::uint32_t top = 0;
            for (const std::uint32_t id : ids) top = std::max(top, id);
            for (std::size_t i = 0; i < prompts.size(); ++i)
                if (ids[i] == 0 && prompts[i].positive) ids[i] = ++top;
            std::vector<std::pair<std::int64_t, std::uint32_t>> strays;   // time point, its background-only object
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                if (ids[i] != 0) continue;
                const Prompt& q = prompts[i];
                const std::vector<std::array<double, 3>> from = q.kind == Prompt::Kind::Scribble ? q.stroke : std::vector<std::array<double, 3>>{q.at};
                std::uint32_t best = 0;
                double bestD = std::numeric_limits<double>::infinity();
                for (std::size_t j = 0; j < prompts.size(); ++j) {
                    if (j == i || ids[j] == 0 || !prompts[j].positive || prompts[j].t != q.t) continue;
                    double d = std::numeric_limits<double>::infinity();
                    for (const std::array<double, 3>& a : from) d = std::min(d, promptDistance(prompts[j], a));
                    if (d < bestD || (d == bestD && ids[j] < best)) {
                        bestD = d;
                        best = ids[j];
                    }
                }
                if (best == 0) {
                    const auto it = std::find_if(strays.begin(), strays.end(), [&](const auto& s) { return s.first == q.t; });
                    if (it != strays.end()) {
                        best = it->second;
                    } else {
                        best = ++top;
                        strays.emplace_back(q.t, best);
                    }
                }
                ids[i] = best;
            }
        }

        // The canonical list of prompts, or invalid_argument naming the entry
        // that is wrong and why. Accepted: the list itself, its JSON text, and
        // the list of JSON texts an older reader of the file made of it.
        ParamValue coercePrompts(const ParamSpec& spec, const json& given) {
            const std::function<std::invalid_argument(const std::string&)> bad = [&](const std::string& what) {
                return std::invalid_argument("parameter '" + spec.key + "': " + what);
            };
            const std::string shapes = R"(points {"x", "y", "z", "t", "label", "object"}, boxes {"kind": "box", "x0", "y0", "z0", "x1", "y1", "z1", "t", "object"} )"
                                       R"(and scribbles {"kind": "scribble", "points": [[x, y, z], ...], "t", "label", "object"})";
            json list = given;
            if (list.is_string()) {
                const std::string text = list.get<std::string>();
                list = text.empty() ? json::array() : json::parse(text, nullptr, false);
                if (list.is_discarded()) throw bad("expected a list of " + shapes + ", got " + given.dump());
            }
            if (list.is_null()) list = json::array();
            if (!list.is_array()) throw bad("expected a list of " + shapes + ", got " + given.dump());
            json out = json::array();
            std::vector<std::uint32_t> ids;   // per entry, 0 until assigned
            for (std::size_t i = 0; i < list.size(); ++i) {
                json e = list[i];
                if (e.is_string()) e = json::parse(e.get<std::string>(), nullptr, false);
                const std::string which = "prompt " + std::to_string(i + 1);
                if (!e.is_object()) throw bad(which + " is not an object: " + list[i].dump() + "; prompts are " + shapes);
                const std::string kind = !e.contains("kind") || e["kind"].is_null() ? std::string("point")
                                         : e["kind"].is_string()                    ? e["kind"].get<std::string>()
                                                                                    : e["kind"].dump();
                json p = {{"kind", kind}};
                if (kind == "point") {
                    for (const char* axis : {"x", "y", "z"}) {
                        if (!e.contains(axis) || !e[axis].is_number()) throw bad(which + " (a point) needs a number '" + axis + "' (voxels)");
                        const double v = e[axis].get<double>();
                        if (!std::isfinite(v) || v < 0.0) throw bad(which + ": '" + axis + "' must be a voxel position >= 0");
                        p[axis] = storedNumber(v);
                    }
                } else if (kind == "box") {
                    for (const char* axis : {"x", "y", "z"}) {
                        const std::string lo = std::string(axis) + "0", hi = std::string(axis) + "1";
                        if (!e.contains(lo) || !e[lo].is_number() || !e.contains(hi) || !e[hi].is_number())
                            throw bad(which + " (a box) needs numbers '" + lo + "' and '" + hi + "' (voxels, " + hi + " exclusive)");
                        const double a0 = e[lo].get<double>(), a1 = e[hi].get<double>();
                        if (!std::isfinite(a0) || !std::isfinite(a1) || a0 < 0.0 || a1 <= a0)
                            throw bad(which + ": a box needs 0 <= " + lo + " < " + hi);
                        p[lo] = storedNumber(a0);
                        p[hi] = storedNumber(a1);
                    }
                } else if (kind == "scribble") {
                    if (!e.contains("points") || !e["points"].is_array() || e["points"].empty())
                        throw bad(which + " (a scribble) needs 'points': [[x, y, z], ...]");
                    json pts = json::array();
                    for (std::size_t k = 0; k < e["points"].size(); ++k)
                        pts.push_back(storedTriple(e["points"][k], bad, which + " point " + std::to_string(k + 1)));
                    p["points"] = std::move(pts);
                } else {
                    throw bad(which + " is a '" + kind + "' prompt; the kinds are point, box and scribble");
                }
                std::int64_t t = 0;
                if (e.contains("t") && !e["t"].is_null()) {
                    const double tv = e["t"].is_number() ? e["t"].get<double>() : -1.0;
                    if (!(tv >= 0.0) || std::floor(tv) != tv) throw bad(which + ": 't' must be a time point index >= 0");
                    t = static_cast<std::int64_t>(tv);
                }
                p["t"] = t;
                int label = 1;
                if (e.contains("label") && !e["label"].is_null()) {
                    const json& l = e["label"];
                    if (l.is_boolean()) label = l.get<bool>() ? 1 : 0;
                    else if (l.is_number() && (l.get<double>() == 0.0 || l.get<double>() == 1.0)) label = static_cast<int>(l.get<double>());
                    else if (l.is_string() && (l.get<std::string>() == "object" || l.get<std::string>() == "background"))
                        label = l.get<std::string>() == "object" ? 1 : 0;
                    else throw bad(which + ": 'label' is 1 (object) or 0 (background), got " + l.dump());
                }
                if (kind == "box") {
                    // the prompt decoder takes a box as "the object in here"
                    if (label != 1) throw bad(which + ": a box always names an object; mark background with a point or a scribble");
                } else {
                    p["label"] = label;
                }
                std::uint32_t id = 0;
                if (e.contains("object") && !e["object"].is_null()) {
                    const json& o = e["object"];
                    const double ov = o.is_number() ? o.get<double>() : 0.0;
                    if (!(ov >= 1.0) || ov > 1e9 || std::floor(ov) != ov)
                        throw bad(which + ": 'object' is the id of the object the prompt belongs to, an integer >= 1, got " + o.dump());
                    id = static_cast<std::uint32_t>(ov);
                }
                ids.push_back(id);
                out.push_back(std::move(p));
            }
            assignPromptObjects(out, ids);
            for (std::size_t i = 0; i < out.size(); ++i) out[i]["object"] = ids[i];
            return ParamJson{out.dump()};
        }
    } // namespace

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
            case ParamType::Prompts: return coercePrompts(spec, j);
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
            case ParamType::Prompts: {
                const json voxel = {{"type", "number"}, {"minimum", 0}};
                const json frame = {{"type", "integer"}, {"minimum", 0}, {"description", "time point (default 0)"}};
                const json label = {{"type", "integer"}, {"enum", {0, 1}}, {"description", "1 object (default), 0 background"}};
                const json object = {{"type", "integer"},
                                     {"minimum", 1},
                                     {"description", "the object it belongs to: one object is one mask, whose label is this id, and a "
                                                     "background prompt refines its object's mask. Default: an object prompt starts a "
                                                     "new object, a background one joins the nearest object on its time point"}};
                s["type"] = "array";
                s["items"] = {{"type", "object"},
                              {"description", "voxels of the step's input, x y z order; one mask per object"},
                              {"properties",
                               {{"kind", {{"type", "string"}, {"enum", {"point", "box", "scribble"}}, {"description", "default point"}}},
                                {"x", voxel},
                                {"y", voxel},
                                {"z", voxel},
                                {"x0", voxel},
                                {"y0", voxel},
                                {"z0", voxel},
                                {"x1", voxel},
                                {"y1", voxel},
                                {"z1", voxel},
                                {"points", {{"type", "array"}, {"items", {{"type", "array"}, {"items", voxel}, {"minItems", 3}, {"maxItems", 3}}}}},
                                {"t", frame},
                                {"label", label},
                                {"object", object}}}};
                desc += " (a point needs x, y, z; a box x0, y0, z0, x1, y1, z1, the upper corner exclusive; a scribble points; the "
                        "prompts with one object id are one object, refined together, at most one box each)";
                break;
            }
        }
        s["description"] = desc;
        return s;
    }

    // --- prompts --------------------------------------------------------------

    Prompt Prompt::point(double x, double y, double z, std::int64_t t, bool positive, std::uint32_t objectId) {
        Prompt p;
        p.kind = Kind::Point;
        p.at = {x, y, z};
        p.t = t;
        p.positive = positive;
        p.objectId = objectId;
        return p;
    }

    Prompt Prompt::boxOf(std::array<double, 6> corners, std::int64_t t, std::uint32_t objectId) {
        Prompt p;
        p.kind = Kind::Box;
        p.box = corners;
        p.t = t;
        p.objectId = objectId;
        return p;
    }

    Prompt Prompt::scribble(std::vector<std::array<double, 3>> points, std::int64_t t, bool positive, std::uint32_t objectId) {
        Prompt p;
        p.kind = Kind::Scribble;
        p.stroke = std::move(points);
        p.t = t;
        p.positive = positive;
        p.objectId = objectId;
        return p;
    }

    std::vector<Prompt> promptsOf(const ParamSet& p, const std::string& key) {
        std::vector<Prompt> out;
        const ParamValue* v = p.find(key);
        if (!v) return out;
        ParamSpec s;
        s.key = key;
        s.type = ParamType::Prompts;
        ParamValue canonical;
        try {
            canonical = coerceToSpec(s, toJson(*v));
        } catch (const std::exception&) {
            return out;   // not a list of prompts: nothing to place, the step's validation says why
        }
        for (const json& e : json::parse(std::get<ParamJson>(canonical).text)) out.push_back(promptOfRecord(e));
        return out;
    }

    ParamValue promptsValue(const std::vector<Prompt>& prompts) {
        json list = json::array();
        for (const Prompt& p : prompts) {
            json e;
            switch (p.kind) {
                case Prompt::Kind::Point:
                    e = {{"kind", "point"}, {"x", p.at[0]}, {"y", p.at[1]}, {"z", p.at[2]}, {"t", p.t}, {"label", p.positive ? 1 : 0}};
                    break;
                case Prompt::Kind::Box:
                    e = {{"kind", "box"}, {"x0", p.box[0]}, {"y0", p.box[1]}, {"z0", p.box[2]}, {"x1", p.box[3]}, {"y1", p.box[4]}, {"z1", p.box[5]}, {"t", p.t}};
                    break;
                case Prompt::Kind::Scribble:
                    e = {{"kind", "scribble"}, {"points", p.stroke}, {"t", p.t}, {"label", p.positive ? 1 : 0}};
                    break;
            }
            if (p.objectId != 0) e["object"] = p.objectId;
            list.push_back(std::move(e));
        }
        ParamSpec s;
        s.key = kPromptsKey;
        s.type = ParamType::Prompts;
        return coerceToSpec(s, list);
    }

    bool isPromptStep(const ParamSet& p) { return p.has(kPromptsKey) && p.getString("task") == kPromptTask; }

    double promptDistance(const Prompt& p, const std::array<double, 3>& q) {
        const auto dist = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
            return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
        };
        switch (p.kind) {
            case Prompt::Kind::Point: return dist(p.at, q);
            case Prompt::Kind::Box: {
                // the box's voxels are [lo, hi - 1] on each axis
                std::array<double, 3> near{};
                for (std::size_t a = 0; a < 3; ++a) near[a] = std::clamp(q[a], p.box[a], std::max(p.box[a], p.box[a + 3] - 1.0));
                return dist(near, q);
            }
            case Prompt::Kind::Scribble: {
                double d = std::numeric_limits<double>::infinity();
                for (const std::array<double, 3>& a : p.stroke) d = std::min(d, dist(a, q));
                return d;
            }
        }
        return std::numeric_limits<double>::infinity();
    }

} // namespace sirius::app
