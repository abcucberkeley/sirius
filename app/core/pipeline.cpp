#include "core/pipeline.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <toml++/toml.hpp>

namespace sirius::app {

    using json = nlohmann::json;

    std::string Step::number(int index) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%02d", index + 1);
        return buf;
    }

    Pipeline::Pipeline() {
        Step load;
        load.id = nextId();
        load.kind = "load";
        load.name = "Load";
        load.pinned = true;
        load.enabled = true;
        load.cache = CachePolicy::Recompute;
        if (const Operation* op = findOperation("load")) load.params = op->defaults();
        steps_.push_back(std::move(load));
    }

    StepId Pipeline::nextId() { return nextId_++; }

    int Pipeline::indexOf(StepId id) const noexcept {
        for (std::size_t i = 0; i < steps_.size(); ++i)
            if (steps_[i].id == id) return static_cast<int>(i);
        return -1;
    }

    const Step* Pipeline::find(StepId id) const noexcept {
        const int i = indexOf(id);
        return i < 0 ? nullptr : &steps_[static_cast<std::size_t>(i)];
    }

    Step* Pipeline::find(StepId id) noexcept {
        const int i = indexOf(id);
        return i < 0 ? nullptr : &steps_[static_cast<std::size_t>(i)];
    }

    int Pipeline::enabledCount() const noexcept {
        return static_cast<int>(std::count_if(steps_.begin(), steps_.end(), [](const Step& s) { return s.enabled; }));
    }

    StepId Pipeline::add(const std::string& kind, int at) {
        const Operation& op = requireOperation(kind);
        Step s;
        s.kind = kind;
        s.name = op.info().name;
        s.cache = op.info().defaultCache;
        s.params = op.defaults();
        return insertStep(std::move(s), at);
    }

    StepId Pipeline::insertStep(Step step, int at) {
        if (step.kind == "load") throw std::invalid_argument("a pipeline has exactly one Load step");
        if (step.id == 0) step.id = nextId();
        else nextId_ = std::max(nextId_, step.id + 1);
        step.pinned = false;
        if (at < 1 || at > size()) at = size();
        const StepId id = step.id;
        steps_.insert(steps_.begin() + at, std::move(step));
        return id;
    }

    void Pipeline::remove(int index) {
        if (index < 1 || index >= size()) return;
        steps_.erase(steps_.begin() + index);
    }

    bool Pipeline::move(int index, int delta) {
        const int j = index + delta;
        if (index < 1 || index >= size() || j < 1 || j >= size() || delta == 0) return false;
        std::swap(steps_[static_cast<std::size_t>(index)], steps_[static_cast<std::size_t>(j)]);
        return true;
    }

    StepId Pipeline::duplicate(int index) {
        if (index < 1 || index >= size()) return 0;
        Step copy = steps_[static_cast<std::size_t>(index)];
        copy.id = 0;
        return insertStep(std::move(copy), index + 1);
    }

    void Pipeline::setEnabled(int index, bool on) {
        if (index < 1 || index >= size()) return;
        steps_[static_cast<std::size_t>(index)].enabled = on;
    }

    void Pipeline::setParams(int index, ParamSet params) {
        if (index < 0 || index >= size()) return;
        Step& s = steps_[static_cast<std::size_t>(index)];
        if (const Operation* op = findOperation(s.kind)) {
            params.applyDefaults(op->info().params);
            params.coerce(op->info().params);
        }
        s.params = std::move(params);
    }

    void Pipeline::setCache(int index, CachePolicy policy) {
        if (index < 0 || index >= size()) return;
        steps_[static_cast<std::size_t>(index)].cache = policy;
    }

    void Pipeline::rename(int index, std::string name) {
        if (index < 0 || index >= size()) return;
        if (name.empty()) name = steps_[static_cast<std::size_t>(index)].op().info().name;
        steps_[static_cast<std::size_t>(index)].name = std::move(name);
    }

    void Pipeline::clearAfterLoad() { steps_.resize(1); }

    void Pipeline::replaceSteps(std::vector<Step> steps, bool keepLoad) {
        Step load = steps_.front();
        steps_.clear();
        auto loadIt = std::find_if(steps.begin(), steps.end(), [](const Step& s) { return s.kind == "load"; });
        if (loadIt != steps.end() && !keepLoad) {
            load.params = loadIt->params;
            load.name = loadIt->name.empty() ? "Load" : loadIt->name;
            if (loadIt->id != 0) load.id = loadIt->id;
        }
        load.enabled = true;
        load.pinned = true;
        steps_.push_back(load);
        // Keep ids that are unique (undo restores snapshots whose ids the
        // executor's cache and the selection refer to); assign fresh ones otherwise.
        for (Step& s : steps) {
            if (s.kind == "load") continue;
            if (s.id != 0 && (s.id == load.id || find(s.id) != nullptr)) s.id = 0;
            insertStep(std::move(s));
        }
    }

    json Pipeline::toJson() const {
        json steps = json::array();
        for (const Step& s : steps_) {
            steps.push_back({{"id", s.id},
                             {"kind", s.kind},
                             {"name", s.name},
                             {"enabled", s.enabled},
                             {"cache", toString(s.cache)},
                             {"params", s.params.toJson()}});
        }
        return {{"version", 1}, {"steps", steps}};
    }

    Pipeline Pipeline::fromJson(const json& j) {
        Pipeline p;
        std::vector<Step> steps;
        if (!j.contains("steps") || !j["steps"].is_array()) throw std::runtime_error("pipeline: missing 'steps'");
        for (const json& sj : j["steps"]) {
            Step s;
            s.kind = sj.value("kind", "");
            const Operation* op = findOperation(s.kind);
            // The Load step is structural: it needs no registered operation
            // (tests and tools may run without the built-ins).
            if (!op && s.kind != "load") throw std::runtime_error("pipeline: unknown operation '" + s.kind + "'");
            s.name = sj.value("name", op ? op->info().name : std::string("Load"));
            s.enabled = sj.value("enabled", true);
            s.cache = cachePolicyFromString(sj.value("cache", "recompute"))
                          .value_or(op ? op->info().defaultCache : CachePolicy::Recompute);
            s.params = ParamSet::fromJson(sj.value("params", json::object()));
            if (op) {
                s.params.applyDefaults(op->info().params);
                s.params.coerce(op->info().params);
            }
            if (sj.contains("id") && sj["id"].is_number_unsigned()) s.id = sj["id"].get<StepId>();
            steps.push_back(std::move(s));
        }
        p.replaceSteps(std::move(steps), false);
        return p;
    }

    // --- TOML ---------------------------------------------------------------------

    namespace {
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
    } // namespace

    void Pipeline::save(const std::string& path) const {
        const json j = toJson();
        toml::table root;
        root.insert("version", j["version"].get<std::int64_t>());
        toml::array steps;
        for (const json& s : j["steps"]) steps.push_back(jsonToToml(s));
        root.insert("steps", steps);
        std::ofstream out(path);
        if (!out) throw std::runtime_error("cannot write pipeline file: " + path);
        out << "# SIRIUS pipeline\n"
            << root << "\n";
    }

    Pipeline Pipeline::load(const std::string& path) {
        toml::table root;
        try {
            root = toml::parse_file(path);
        } catch (const toml::parse_error& e) {
            throw std::runtime_error("cannot parse pipeline file " + path + ": " + std::string(e.description()));
        }
        return fromJson(tomlToJson(root));
    }

    std::string Pipeline::toPythonScript(const std::string& datasetPath) const {
        std::ostringstream py;
        py << "# Generated by SIRIUS: reproduces the workbench pipeline with the sirius Python package.\n"
              "import json\n"
              "import numpy as np\n"
              "import sirius\n"
              "from sirius.workbench import run_pipeline\n\n"
              "DATASET = "
           << json(datasetPath).dump() << "\n"
                                          "PIPELINE = json.loads(r'''"
           << toJson().dump(2) << "''')\n\n"
                                  "if __name__ == '__main__':\n"
                                  "    result, meta = run_pipeline(DATASET, PIPELINE)\n"
                                  "    print('result', result.shape, meta)\n";
        return py.str();
    }

    Pipeline Pipeline::example() {
        Pipeline p;
        auto addIf = [&](const char* kind, bool enabled, CachePolicy cache) {
            if (!findOperation(kind)) return;
            const StepId id = p.add(kind);
            p.setEnabled(p.indexOf(id), enabled);
            p.setCache(p.indexOf(id), cache);
        };
        addIf("sim", true, CachePolicy::Disk);
        addIf("deskew", false, CachePolicy::Recompute);
        addIf("einsum", true, CachePolicy::Memory);
        addIf("contrast", true, CachePolicy::Recompute);
        addIf("decon", false, CachePolicy::Disk);
        addIf("merge", true, CachePolicy::Recompute);
        addIf("seg", true, CachePolicy::Disk);
        addIf("volrec", true, CachePolicy::Memory);
        return p;
    }

} // namespace sirius::app
