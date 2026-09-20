#ifndef SIRIUS_APP_OPS_TORCH_MODEL_HPP
#define SIRIUS_APP_OPS_TORCH_MODEL_HPP

// What the Python worker says about a segmentation model, and the one line
// the parameter panel and the model hub show for it.

#include <string>

#include <nlohmann/json.hpp>

#include "core/rpc.hpp"

namespace sirius::app {

    // "TorchScript · in (1, 1, Z, Y, X) float32 · out (1, 3, Z, Y, X) · 41 MB"
    // from the worker's model_info; throws when the worker cannot load it.
    nlohmann::json torchModelInfo(RemoteWorker& worker, const std::string& modelPath);
    std::string torchModelSummary(const nlohmann::json& info);

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_TORCH_MODEL_HPP
