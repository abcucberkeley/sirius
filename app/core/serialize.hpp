#ifndef SIRIUS_APP_SERIALIZE_HPP
#define SIRIUS_APP_SERIALIZE_HPP

// What a step's result is made of besides its array, as the wire carries it
// between the application and the engine on a cluster node
// (core/engine_server.hpp): JSON for the structure, tensors (core/rpc.hpp)
// for the numbers that are many. Every form round-trips: x == fromJson(toJson(x)).
//
//   DatasetMeta   JSON (dims, pixel type, voxel size, channels, SIM layout, tiles)
//   StepReport    JSON {id, index, state, seconds, note, error}
//   Lineage       JSON {"child": parent, ...} (lineageFromJson in core/tracks.hpp reads it)
//   Diagnostics   JSON with every image's values in a float32 tensor of its own
//                 ("<prefix>img<i>", (rows, cols)); decodeDiagnostics takes them back

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/dataset.hpp"
#include "core/diagnostics.hpp"
#include "core/executor.hpp"
#include "core/label_frames.hpp"
#include "core/rpc.hpp"

namespace sirius::app {

    // --- DatasetMeta ---------------------------------------------------------------------
    nlohmann::json toJson(const DatasetMeta& meta);
    // Throws std::runtime_error on a value of the wrong kind; missing fields keep their defaults.
    DatasetMeta datasetMetaFromJson(const nlohmann::json& j);

    // The protocol's dtype name of a pixel type ("uint16") and back
    // (Float32 for a name it does not know).
    const char* dtypeName(PixelType t) noexcept;
    PixelType pixelTypeFromName(const std::string& dtype) noexcept;

    // --- StepReport ----------------------------------------------------------------------
    nlohmann::json toJson(const StepReport& report);
    StepReport stepReportFromJson(const nlohmann::json& j);
    std::optional<StepReport::State> stepStateFromString(const std::string& s) noexcept;

    // --- Lineage -------------------------------------------------------------------------
    nlohmann::json lineageToJson(const Lineage& lineage);

    // --- Diagnostics ---------------------------------------------------------------------
    struct EncodedDiagnostics {
        nlohmann::json json;
        std::vector<rpc::Tensor> tensors;
    };
    // `prefix` keeps the tensor names of several steps' diagnostics apart in one frame.
    EncodedDiagnostics encodeDiagnostics(const Diagnostics& d, const std::string& prefix = {});
    // Throws ProtocolError when an image's tensor is missing or does not match its size.
    Diagnostics decodeDiagnostics(const nlohmann::json& j, const std::vector<rpc::Tensor>& tensors);

} // namespace sirius::app

#endif // SIRIUS_APP_SERIALIZE_HPP
