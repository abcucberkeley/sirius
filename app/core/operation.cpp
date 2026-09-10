#include "core/operation.hpp"
#include "core/cancel.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/array_source.hpp"

namespace sirius::app {

    const char* toString(Backend b) noexcept {
        switch (b) {
            case Backend::Cuda: return "CUDA";
            case Backend::Cpu: return "CPU";
            case Backend::Hpc: return "HPC";
        }
        return "?";
    }

    std::optional<Backend> backendFromString(const std::string& s) noexcept {
        std::string l;
        for (char c : s) l += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (l == "cuda" || l == "gpu") return Backend::Cuda;
        if (l == "cpu") return Backend::Cpu;
        if (l == "hpc" || l == "slurm" || l == "remote") return Backend::Hpc;
        return std::nullopt;
    }

    const char* toString(CachePolicy c) noexcept {
        switch (c) {
            case CachePolicy::Memory: return "memory";
            case CachePolicy::Disk: return "disk";
            case CachePolicy::Recompute: return "recompute";
        }
        return "?";
    }

    std::optional<CachePolicy> cachePolicyFromString(const std::string& s) noexcept {
        std::string l;
        for (char c : s) l += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (l == "memory" || l == "ram" || l == "m") return CachePolicy::Memory;
        if (l == "disk" || l == "d") return CachePolicy::Disk;
        if (l == "recompute" || l == "none" || l == "r") return CachePolicy::Recompute;
        return std::nullopt;
    }

    void StepContext::throwIfCancelled() const {
        if (isCancelled()) throw CancelledError{};
    }

    // --- StepInput --------------------------------------------------------------

    ArrayPtr StepInput::materialize(const std::function<void(double, const std::string&)>& progress) const {
        if (hasArray()) return array;
        if (!source) throw std::runtime_error("step input has neither an array nor a source");
        return source->readAll(progress);
    }

    Buffer<float> StepInput::readVolume(Index c, Index t) const {
        const Dims5& d = meta.dims;
        if (c < 0 || c >= d.c || t < 0 || t >= d.t)
            throw std::out_of_range("readVolume: (c " + std::to_string(c) + ", t " + std::to_string(t) +
                                    ") outside " + d.toString());
        Buffer<float> out(Shape{d.z, d.y, d.x});
        if (hasArray()) {
            copy(array->volume(c, t), out);
        } else if (source) {
            source->readVolume(c, t, out.data());
        } else {
            throw std::runtime_error("step input has neither an array nor a source");
        }
        return out;
    }

    // --- Operation defaults -----------------------------------------------------

    std::string Operation::summary(const ParamSet& params, const DatasetMeta&) const {
        std::string out;
        int n = 0;
        for (const ParamSpec& s : info().params) {
            if (s.advanced || s.readOnly) continue;
            const ParamValue* v = params.find(s.key);
            if (!v) continue;
            if (n++) out += " · ";
            std::string label = s.label;
            std::transform(label.begin(), label.end(), label.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            out += label + " " + toDisplayString(*v);
            if (!s.unit.empty()) out += " " + s.unit;
            if (n >= 3) break;
        }
        return out.empty() ? std::string("default parameters") : out;
    }

    Validation Operation::validate(const ParamSet& params, const DatasetMeta& input) const {
        Validation v;
        if (input.dims.numel() <= 0) v.errors.push_back("No input data.");
        for (const ParamSpec& s : info().params) {
            if (s.type != ParamType::Channel) continue;
            // a parameter the current settings hide does not apply (the
            // moving channel of a time alignment): a single-channel movie
            // could not be time-aligned because "channel 1 does not exist"
            if (!s.visibleFor(params)) continue;
            const std::int64_t c = params.getInt(s.key, 0);
            if (c < 0 || c >= input.dims.c)
                v.errors.push_back(s.label + ": channel " + std::to_string(c) + " does not exist (input has " +
                                   std::to_string(input.dims.c) + ").");
        }
        return v;
    }

    DatasetMeta Operation::outputMeta(const ParamSet&, const DatasetMeta& input) const { return input; }

    std::size_t Operation::estimatedOutputBytes(const ParamSet& params, const DatasetMeta& input) const {
        return outputMeta(params, input).dims.bytes();
    }

    // --- registry -----------------------------------------------------------------

    namespace {
        struct Registry {
            std::mutex mutex;
            std::vector<std::unique_ptr<Operation>> ops;
            std::map<std::string, Operation*> byKind;
        };
        Registry& registry() {
            static Registry r;
            return r;
        }
    } // namespace

    void registerOperation(std::unique_ptr<Operation> op) {
        if (!op) return;
        Registry& r = registry();
        std::lock_guard<std::mutex> g(r.mutex);
        const std::string kind = op->kind();
        auto it = std::find_if(r.ops.begin(), r.ops.end(), [&](const auto& o) { return o->kind() == kind; });
        if (it != r.ops.end()) {
            *it = std::move(op);
            r.byKind[kind] = it->get();
        } else {
            r.byKind[kind] = op.get();
            r.ops.push_back(std::move(op));
        }
    }

    const Operation* findOperation(const std::string& kind) noexcept {
        Registry& r = registry();
        std::lock_guard<std::mutex> g(r.mutex);
        auto it = r.byKind.find(kind);
        return it == r.byKind.end() ? nullptr : it->second;
    }

    const Operation& requireOperation(const std::string& kind) {
        const Operation* op = findOperation(kind);
        if (!op) throw std::out_of_range("unknown operation kind '" + kind + "'");
        return *op;
    }

    std::vector<const Operation*> allOperations() {
        Registry& r = registry();
        std::lock_guard<std::mutex> g(r.mutex);
        std::vector<const Operation*> out;
        for (const auto& o : r.ops) out.push_back(o.get());
        return out;
    }

    std::vector<std::pair<std::string, std::vector<const Operation*>>> operationGroups() {
        static const char* order[] = {"Reconstruct", "Reduce", "Intensity", "Geometry", "Combine", "Segment", "User", "Input"};
        std::vector<std::pair<std::string, std::vector<const Operation*>>> groups;
        for (const char* g : order) {
            std::vector<const Operation*> ops;
            for (const Operation* op : allOperations())
                if (op->info().group == g && op->kind() != "load") ops.push_back(op);
            if (!ops.empty()) groups.emplace_back(g, std::move(ops));
        }
        // anything in a group we did not list
        for (const Operation* op : allOperations()) {
            if (op->kind() == "load") continue;
            bool placed = false;
            for (auto& g : groups)
                if (g.first == op->info().group) placed = true;
            if (!placed) {
                auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.first == op->info().group; });
                if (it == groups.end()) groups.emplace_back(op->info().group, std::vector<const Operation*>{op});
                else it->second.push_back(op);
            }
        }
        return groups;
    }

    // --- schema export ---------------------------------------------------------------

    namespace {
        const char* paramTypeName(ParamType t) noexcept {
            switch (t) {
                case ParamType::Bool: return "bool";
                case ParamType::Int: return "int";
                case ParamType::Double: return "double";
                case ParamType::String: return "string";
                case ParamType::Path: return "path";
                case ParamType::Choice: return "choice";
                case ParamType::Channel: return "channel";
                case ParamType::Axes: return "axes";
                case ParamType::DoubleList: return "double_list";
                case ParamType::StringList: return "string_list";
            }
            return "?";
        }
    } // namespace

    nlohmann::json operationSchemas() {
        registerBuiltinOperations();
        nlohmann::json ops = nlohmann::json::array();
        for (const Operation* op : allOperations()) {
            const OpInfo& info = op->info();
            nlohmann::json params = nlohmann::json::array();
            for (const ParamSpec& s : info.params) {
                nlohmann::json p = {
                    {"key", s.key},
                    {"label", s.label},
                    {"type", paramTypeName(s.type)},
                    {"default", toJson(s.defaultValue)},
                    {"advanced", s.advanced},
                    {"read_only", s.readOnly},
                };
                if (s.type == ParamType::Choice) p["choices"] = s.choices;
                if (std::isfinite(s.min)) p["min"] = s.min;
                if (std::isfinite(s.max)) p["max"] = s.max;
                if (!s.unit.empty()) p["unit"] = s.unit;
                if (!s.help.empty()) p["help"] = s.help;
                // when the step reads this at all: a caller that knows the
                // rules can avoid setting something the settings ignore
                if (!s.visibility.empty()) {
                    nlohmann::json rules = nlohmann::json::array();
                    for (const ParamSpec::Visibility& v : s.visibility)
                        rules.push_back({{"key", v.key}, {"values", v.values}, {"negate", v.negate}});
                    p["visible_when"] = std::move(rules);
                }
                params.push_back(std::move(p));
            }
            ops.push_back({
                {"kind", info.kind},
                {"name", info.name},
                {"group", info.group},
                {"plugin", info.plugin},
                {"produces_labels", info.producesLabels},
                {"needs_labels", info.needsLabels},
                {"remote_capable", info.remoteCapable},
                {"params", std::move(params)},
            });
        }
        return {{"version", 1}, {"operations", std::move(ops)}};
    }

} // namespace sirius::app
