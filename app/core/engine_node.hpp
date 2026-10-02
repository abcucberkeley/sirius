#ifndef SIRIUS_APP_ENGINE_NODE_HPP
#define SIRIUS_APP_ENGINE_NODE_HPP

// What SIRIUS's engine on a cluster node computes and holds
// (core/engine_server.hpp serves it): the application's pipelines, run by the
// same Executor::run the application runs, with the outputs kept here and
// named by handles the application draws through dataset_view / _read /
// _stats (core/remote_source.hpp, NodeOutputSource). The application stays the
// authority over the pipeline, its undo and its validation; the node only
// computes and holds data.
//
//   pipeline_run   {pipeline, target, device, hub_token?, have?}
//                  progress {fraction, message, step, state}; result
//                  {session, device, reports: [StepReport], outputs: [{index,
//                  step_id, handle, fingerprint, held, meta, note, seconds,
//                  cache, bytes, ran_on: {backend, device}, labels: {present,
//                  count}, diagnostics?}], error?} + the diagnostics' image
//                  tensors. A step that fails ends the run with "error" and
//                  the outputs made before it; a cancelled run is "cancelled".
//   step_preview   {pipeline, index, params?, initial?} -> {diagnostics | null,
//                  initial_params?}: the operation's preview on its input as
//                  the node holds it (a Load input is opened for it)
//   step_validate  {pipeline, index} -> {errors, warnings}, with the node's files
//   output_stats   {path (a handle or a dataset path), options?, statistics}
//                  -> {channels} (core/statistics.hpp, computed here)
//   put_file       {key, name, size, offset} + tensor "data" (uint8) -> {received,
//                  path?}: a file of the application's, uploaded in chunks
//                  into the node scratch (only after the user agreed to it)
//   stat_file      {key, name, size} -> {exists, path?}
//   outputs_release {handles} -> {released};  cache_status {} -> {steps, bytes}
//
// Paths: the application's pipeline names the cluster's files as
// "cluster://<host>/<absolute path>"; each such parameter is the absolute
// path here. Fingerprints are the node's own (its files' stamps), so each
// side keys its cache by its own fingerprint; the application stores the
// handle. Handles carry the engine's session: another engine process (a new
// job) holds none of them, and says so.

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "core/dataset_service.hpp"
#include "core/rpc.hpp"
#include "core/rpc_server.hpp"

namespace sirius::app {

    class Pipeline;

    // The pipeline as the node runs it: every "cluster://<host>/<path>" in a
    // Path or StringList parameter becomes <path>.
    nlohmann::json nodePipelineJson(const nlohmann::json& pipeline);

    class EngineNode {
    public:
        struct Options {
            std::filesystem::path scratch;      // the node's scratch; "" = a temporary directory of its own
            std::string defaultDevice = "auto";  // what "cuda" means without a GPU: the engine's --device
            std::uint64_t maxUploadBytes = std::uint64_t{1} << 40;
            // A connection to the Python worker child for a step that needs it
            // (seg, plugins), and back when the run is done.
            std::function<std::unique_ptr<RemoteWorker>(const std::function<bool()>& cancelled)> takePython;
            std::function<void(std::unique_ptr<RemoteWorker>)> giveBackPython;
            std::function<void(const std::string&)> log;
        };

        explicit EngineNode(Options options);
        ~EngineNode();   // removes the scratch it made, and the uploads
        EngineNode(const EngineNode&) = delete;
        EngineNode& operator=(const EngineNode&) = delete;

        // This process's session, the first part of every handle it gives out.
        const std::string& session() const noexcept;
        const std::filesystem::path& scratch() const noexcept;
        // The output a handle names, for dataset_* (throws DatasetError).
        DatasetService::ResolvedOutput resolve(const std::string& handle) const;
        // {"steps", "bytes"} held now.
        nlohmann::json cacheStatus() const;

        rpc::Reply pipelineRun(const rpc::Request& req, rpc::CallContext& ctx);
        rpc::Reply stepPreview(const rpc::Request& req, rpc::CallContext& ctx);
        rpc::Reply stepValidate(const rpc::Request& req);
        rpc::Reply outputStats(const rpc::Request& req, rpc::CallContext& ctx);
        rpc::Reply putFile(const rpc::Request& req);
        rpc::Reply statFile(const rpc::Request& req);
        rpc::Reply releaseOutputs(const rpc::Request& req);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_ENGINE_NODE_HPP
