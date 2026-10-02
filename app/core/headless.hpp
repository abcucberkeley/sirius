#ifndef SIRIUS_APP_HEADLESS_HPP
#define SIRIUS_APP_HEADLESS_HPP

// A Workbench without a window, driven by tool calls: what sirius-cli's one-shot commands,
// session and MCP server share. One thread (the caller's) owns the workbench; runs execute
// on a thread of its own and are folded back by pump().
//
// The tool table is the ToolApi's (the tools the GUI assistant uses) with the view tools
// taken out, some replaced by headless versions and the rest added: datasets, pipelines
// as files, runs that are waited for or polled, renders, statistics, exports and the
// Python worker. A tool call has failed exactly when ToolApi's result carries
// "error_kind"; everything else is a value, even with an "error" key in it.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/agent_protocol.hpp"
#include "core/array_source.hpp"
#include "core/display_model.hpp"
#include "core/export.hpp"
#include "core/local_worker.hpp"
#include "core/tool_api.hpp"
#include "core/workbench.hpp"

namespace sirius::app {
    struct HeadlessOptions {
        std::filesystem::path scratchDir;                 // exists; executor disk cache, renders/
        std::string python;                               // --python
        std::string workerDir;                            // --worker-dir
        enum class Plugins { Auto,
                             On,
                             Off } plugins = Plugins::Auto;
        std::string backend = "auto";                     // auto | cpu | cuda | hpc
        int cudaDevice = 0;                               // Workbench::kAllCudaDevices = all
        std::string hpcDevice = "gpu";                    // gpu | cpu: the HPC worker's (Workbench::hpcDevice)
        std::optional<RemoteConfig> hpc;                  // --hpc; the only endpoint tools may select (D27)
        std::string hubToken;                             // $HF_TOKEN
        std::string recordPath;
        bool allowWorkerSetup = false;                    // setup_worker_env may download
        bool readOnly = false;                            // tools with destructive hints are not listed
        bool allowNetworkPaths = false;                   // --allow-network-paths: tools may name \\server\share

        std::string createdBy = "sirius-cli";             // pyenv marker
        std::string setupHint = "Run `sirius-cli worker setup --yes` to create SIRIUS's own Python environment "
                                "(downloads numpy from pypi.org; an agent should ask the user first), "
                                "or pass --python <an interpreter that has numpy>.";
        // Every workbench and worker log line, on the main thread (D32); the CLI host writes stderr.
        std::function<void(const std::string& source, const std::string& line)> logSink;
    };
    class HeadlessWorkbench final : public agent::ToolDispatcher {
    public:
        // Throws ToolFailure when the options cannot be honoured: a backend this machine
        // does not have (cuda without a GPU, hpc without an endpoint), a record file that
        // cannot be written.
        explicit HeadlessWorkbench(HeadlessOptions options);
        ~HeadlessWorkbench() override;                    // cancels and joins a run, stops the worker
        Workbench& workbench() noexcept;
        ToolApi& toolApi() noexcept;
        LocalWorker& worker() noexcept;
        const std::string& workspaceId() const noexcept;  // "ws_" + 12 hex
        std::vector<agent::ToolDescriptor> tools() const override;
        bool hasTool(const std::string& name) const override;
        agent::ToolResult call(const std::string& name, const nlohmann::json& args, const agent::CallContext& ctx) override;
        agent::Status status() const override;
        void cancelActive() override;
        void pump() override;
        std::vector<nlohmann::json> takeEvents() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    // Shared helpers (also used by tests and the CLI). They throw ToolFailure.
    nlohmann::json datasetInfo(const DatasetMeta& meta, const OpenResult* opened = nullptr);   // DatasetInfo, section 3.5
    OpenOptions openOptionsFromJson(const nlohmann::json& args);
    ExportOptions exportOptionsFromJson(const nlohmann::json& args, const DatasetMeta& meta);
    struct RenderRequest {
        int step = -1;                                    // 0-based; -1 = default rule
        std::string plane = "xy";                         // xy | xz | yz | mip
        std::vector<Index> z;                             // empty = middle; several = grid (xy)
        Index t = 0;
        std::optional<Index> y, x;
        std::vector<Index> channels;                      // empty = all
        std::string layout = "blend";                     // blend | channels
        std::string window = "auto";                      // auto | full
        struct ChannelWindow {
            Index channel = 0;
            float lo = 0, hi = 1, gamma = 1;
        };
        std::vector<ChannelWindow> windows;
        std::optional<bool> labels;
        double labelOpacity = 0.45;
        std::uint32_t label = 0;
        bool solo = false;
        std::array<int, 4> region{0, 0, 0, 0};            // x, y, w, h; w == 0 = whole plane
        int maxSize = 1024;                               // <= 1568; 0 = native (still capped)
        bool physicalZ = true;
        std::string format;                               // "" = png with the JPEG fallback; "png" | "jpeg"
        std::size_t maxBytes = std::size_t{4} << 20;      // coarsen, then too_large (D18)
    };
    struct RenderResult {
        std::vector<std::uint8_t> bytes;
        std::string mimeType;
        int width = 0, height = 0, factor = 1;
        nlohmann::json caption;                           // section 3.9 (without path; the caller adds it)
    };
    RenderRequest renderRequestFromJson(const nlohmann::json& args);
    RenderResult renderOutput(std::shared_ptr<const StepOutput> out, int stepIndex, const RenderRequest& r,
                              display::DisplayModel& model, const std::function<bool()>& cancelled = {});

    // The pure halves of some tools, for the same callers. Like the helpers above they
    // throw ToolFailure.
    //
    // A help page is named by a file stem in the help directory: [A-Za-z0-9_-]+ and
    // nothing else, so no name can reach outside it.
    bool isHelpPageName(const std::string& name) noexcept;
    // {page, title, path, exists, markdown, truncated}, the Markdown cut at a UTF-8
    // boundary after `maxChars`. not_found when there is no such page.
    nlohmann::json helpPageJson(const std::string& page, std::size_t maxChars = 60000);
    nlohmann::json helpPageList();                                          // {pages:[{page, title}]}
    // One entry of list_operations: {kind, name, group, params, presets, produces_labels,
    // needs_labels, needs_worker, plugin, gpu}; `params` is a count, or the parameters with `detail`.
    nlohmann::json operationJson(const Operation& op, bool detail);
    nlohmann::json describeOperation(const Operation& op);                 // describe_operation
    // get_diagnostics: the summary, table, facts, curves, histograms, images and tabs of a
    // step's diagnostics; `detail` adds the curves' points (at most 200) and the histograms' bins.
    nlohmann::json diagnosticsJson(const Diagnostics& d, int stepIndex, bool detail);
    // A diagnostics image as the diagnostics panel draws it (a robust grey window, the
    // marks on top), reduced to at most `maxSize` (capped at 1568) on its longer side;
    // always a PNG, one grey channel when there are no marks. Over `maxBytes` it is
    // reduced up to three more times, then too_large.
    RenderResult renderDiagnosticImage(const DiagnosticImage& image, int maxSize, std::size_t maxBytes = std::size_t{4} << 20);
    // export_result's writing half, as File > Export result does it: the pipeline
    // sidecar (options.includePipeline, written with Pipeline::save), a copy of the
    // labels, then the pixels. `labelsOnly` writes the labels alone, as one 32-bit TIFF.
    // {path, format, dtype, shape, files, bytes, seconds, warnings}; export_failed,
    // cancelled (CancelledError) or invalid_argument otherwise.
    nlohmann::json exportStepOutput(std::shared_ptr<const StepOutput> out, const Pipeline& pipeline, const ExportOptions& options,
                                    bool labelsOnly, const std::function<void(double, const std::string&)>& progress = {},
                                    const std::function<bool()>& cancelled = {});
} // namespace sirius::app

#endif // SIRIUS_APP_HEADLESS_HPP
