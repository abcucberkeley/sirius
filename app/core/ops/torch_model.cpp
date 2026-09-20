// What the worker reports about a model, and its one-line summary. Apart from
// the segmentation step that runs such a model: the parameter panel and the
// model hub dialog ask for this before any step exists.
#include "core/ops/torch_model.hpp"

#include <cstdint>
#include <string>

#include "core/ops/common.hpp"   // formatBytes

namespace sirius::app {

    namespace {

        std::string shapeText(const nlohmann::json& shape) {
            if (!shape.is_array()) return "?";
            std::string out = "(";
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i) out += ", ";
                const nlohmann::json& e = shape[i];
                if (e.is_number()) out += std::to_string(e.get<long long>());
                else if (e.is_string()) out += e.get<std::string>();
                else out += "?";
            }
            return out + ")";
        }

    } // namespace

    nlohmann::json torchModelInfo(RemoteWorker& worker, const std::string& modelPath) {
        // "spec" carries hub / family specs; "path" / "model" keep older workers working
        WorkerResult r = worker.call("model_info", {{"path", modelPath}, {"model", modelPath}, {"spec", modelPath}});
        return r.result;
    }

    std::string torchModelSummary(const nlohmann::json& info) {
        if (!info.is_object()) return "no model";
        std::string out = info.value("format", "TorchScript");
        if (info.contains("available") && info["available"].is_boolean()) {
            // a model family (cellpose, micro-sam) or an hf: file not downloaded yet
            if (info.value("model", std::string()).size()) out += " " + info.value("model", std::string());
            if (!info["available"].get<bool>()) {
                const std::string hint = info.value("install_hint", std::string());
                return out + " · not installed (Hub… installs it" + (hint.empty() ? ")" : ": " + hint + ")");
            }
            if (info.contains("cached") && info["cached"].is_boolean() && !info["cached"].get<bool>())
                return out + " " + info.value("repo", std::string()) + " · downloads on first run";
            if (!info.contains("input_shape")) {
                if (info.contains("version") && info["version"].is_string() && !info["version"].get<std::string>().empty())
                    out = info.value("format", std::string()) + " " + info["version"].get<std::string>() + " " + info.value("model", std::string());
                out += " · returns labels";
                if (info.contains("weights_cached") && info["weights_cached"].is_boolean())
                    out += info["weights_cached"].get<bool>() ? " · weights cached" : " · weights download on first run";
                if (info.contains("warning") && info["warning"].is_string()) out += " · " + info["warning"].get<std::string>();
                return out;
            }
        }
        if (info.contains("input_shape")) {
            out += " · in " + shapeText(info["input_shape"]);
            if (info.contains("input_dtype")) out += " " + info["input_dtype"].get<std::string>();
        }
        if (info.contains("output_shape")) out += " · out " + shapeText(info["output_shape"]);
        if (info.contains("size_bytes") && info["size_bytes"].is_number())
            out += " · " + formatBytes(info["size_bytes"].get<std::uint64_t>());
        return out;
    }

} // namespace sirius::app
