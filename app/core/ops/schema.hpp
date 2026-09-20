#ifndef SIRIUS_APP_OPS_SCHEMA_HPP
#define SIRIUS_APP_OPS_SCHEMA_HPP

// The parameter specs of every operation as JSON: what the Python mirror is
// checked against. Above the registry, not part of it -- naming every
// built-in is what it is for.

#include <nlohmann/json_fwd.hpp>

namespace sirius::app {

    // Every registered operation (built-ins first) with its parameter specs
    // as JSON: {"version", "operations": [{kind, name, group, plugin,
    // produces_labels, needs_labels, params: [{key, label, type, default,
    // choices, min, max, unit, advanced}]}]}. The Python mirror
    // (bindings/python/sirius/workbench.py) is checked against a committed
    // snapshot of it (bindings/python/sirius/op_schema.json), so a parameter
    // renamed here fails that test instead of silently changing a pipeline.
    nlohmann::json operationSchemas();

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_SCHEMA_HPP
