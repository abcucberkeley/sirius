#include "core/ops/schema.hpp"

#include <cmath>

#include <nlohmann/json.hpp>

#include "core/ops/builtin.hpp"

namespace sirius::app {

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
