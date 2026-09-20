// The operation schema export (operationSchemas): every built-in kind with
// its parameter keys, types, defaults and choices as JSON. The Python mirror
// of the operations (bindings/python/sirius/workbench.py) is checked against
// a committed snapshot of this JSON, bindings/python/sirius/op_schema.json,
// by bindings/tests/test_workbench_schema.py. Regenerate the snapshot with
//
//     SIRIUS_OP_SCHEMA_OUT=bindings/python/sirius/op_schema.json sirius_tests "[schema]"
//
// whenever a parameter is added or renamed; without the variable the test
// only checks the export is well formed.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

#include <nlohmann/json.hpp>

#include "core/operation.hpp"
#include "core/ops/schema.hpp"
#include "core/ops/builtin.hpp"

using namespace sirius::app;

TEST_CASE("operation schemas list every built-in kind with typed parameters", "[app][schema]") {
    registerBuiltinOperations();
    const nlohmann::json schema = operationSchemas();
    REQUIRE(schema.is_object());
    REQUIRE(schema.at("version").get<int>() == 1);
    const nlohmann::json& ops = schema.at("operations");
    REQUIRE(ops.is_array());

    std::set<std::string> kinds;
    for (const nlohmann::json& op : ops) {
        const std::string kind = op.at("kind").get<std::string>();
        REQUIRE(kinds.insert(kind).second);   // one entry per kind
        REQUIRE(op.at("params").is_array());
        std::set<std::string> keys;
        for (const nlohmann::json& p : op.at("params")) {
            REQUIRE(p.at("key").is_string());
            REQUIRE(keys.insert(p.at("key").get<std::string>()).second);
            const std::string type = p.at("type").get<std::string>();
            REQUIRE(type != "?");
            REQUIRE(p.contains("default"));
            if (type == "choice") {
                REQUIRE(p.at("choices").is_array());
                REQUIRE(!p.at("choices").empty());
                // the default is one of the choices
                bool found = false;
                for (const nlohmann::json& c : p.at("choices"))
                    if (c == p.at("default")) found = true;
                REQUIRE(found);
            }
        }
    }
    for (auto factory : builtinOperationFactories()) REQUIRE(kinds.count(factory()->kind()) == 1);
    // the kinds the Python mirror is written against
    for (const char* k : {"einsum", "maxproj", "meant", "contrast", "flatfield", "bleach", "croppad", "resample",
                          "merge", "threshold", "classic", "cleanup", "seg", "sim", "load"})
        REQUIRE(kinds.count(k) == 1);

    if (const char* out = std::getenv("SIRIUS_OP_SCHEMA_OUT"); out && *out) {
        std::ofstream f(out);
        REQUIRE(f.good());
        f << schema.dump(2) << '\n';
        REQUIRE(f.good());
    }
}

TEST_CASE("no parameter label or help carries a line break or a run of spaces", "[app][schema]") {
    // These strings go straight into a tooltip and into op_schema.json. A
    // help text split across two source lines picks up the continuation's
    // indentation as content unless the literals are simply adjacent, and
    // the result is a newline and thirty spaces in the middle of a sentence
    // -- which is exactly what happened to the SIM pattern-angle help. The
    // compiler cannot see the difference, so the tests do.
    registerBuiltinOperations();
    const nlohmann::json schema = operationSchemas();
    int checked = 0;
    for (const nlohmann::json& op : schema.at("operations")) {
        const std::string kind = op.value("kind", std::string());
        for (const nlohmann::json& p : op.value("params", nlohmann::json::array())) {
            const std::string key = p.value("key", std::string());
            for (const char* field : {"label", "help", "unit"}) {
                if (!p.contains(field) || !p[field].is_string()) continue;
                const std::string text = p[field].get<std::string>();
                INFO(kind << '.' << key << '.' << field << " = " << text);
                CHECK(text.find('\n') == std::string::npos);
                CHECK(text.find('\t') == std::string::npos);
                CHECK(text.find("  ") == std::string::npos);
                ++checked;
            }
        }
    }
    CHECK(checked > 100);   // the sweep actually looked at something
}
