#include "core/workbench.hpp"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <limits>
#include <ctime>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <tuple>

#include "core/build_info.hpp"
#include "core/cancel.hpp"
#include "core/ops/load.hpp"
#include "core/ops/plugin.hpp"
#include "core/ops/builtin.hpp"
#include "core/remote_source.hpp"
#include "core/serialize.hpp"
#include "core/sha256.hpp"

namespace sirius::app {

    using json = nlohmann::json;

    // --- enums ------------------------------------------------------------------

    const char* toString(ViewMode m) noexcept {
        switch (m) {
            case ViewMode::Ortho: return "ortho";
            case ViewMode::Volume: return "3d";
            case ViewMode::Compare: return "compare";
        }
        return "?";
    }
    const char* toString(ViewerTool t) noexcept {
        switch (t) {
            case ViewerTool::Navigate: return "nav";
            case ViewerTool::Probe: return "probe";
            case ViewerTool::Measure: return "measure";
            case ViewerTool::Roi: return "roi";
            case ViewerTool::Paint: return "paint";
            case ViewerTool::Prompt: return "prompt";
        }
        return "?";
    }
    const char* toString(PaintTool t) noexcept {
        switch (t) {
            case PaintTool::Brush: return "brush";
            case PaintTool::Erase: return "erase";
            case PaintTool::Fill: return "fill";
            case PaintTool::Pick: return "pick";
            case PaintTool::Merge: return "merge";
            case PaintTool::Split: return "split";
            case PaintTool::Delete: return "delete";
            case PaintTool::Lasso: return "lasso";
        }
        return "?";
    }

    const char* toString(PromptMode m) noexcept {
        switch (m) {
            case PromptMode::Box: return "box";
            case PromptMode::Click: return "click";
            case PromptMode::Scribble: return "scribble";
        }
        return "?";
    }

    namespace {
        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        // A worker message and the host's next step as one text:
        // "<message>. <hint>". The message ends in a full stop either way,
        // as the other log lines do.
        std::string withHint(std::string message, const std::string& hint) {
            if (!message.empty() && message.back() != '.' && message.back() != '!' && message.back() != '?') message += '.';
            if (!hint.empty()) message += (message.empty() ? "" : " ") + hint;
            return message;
        }


        std::string bytesText(std::uint64_t bytes) {
            char buf[32];
            if (bytes >= 1000000000ull) std::snprintf(buf, sizeof buf, "%.1f GB", static_cast<double>(bytes) / 1e9);
            else if (bytes >= 1000000ull) std::snprintf(buf, sizeof buf, "%.1f MB", static_cast<double>(bytes) / 1e6);
            else std::snprintf(buf, sizeof buf, "%.0f kB", static_cast<double>(bytes) / 1e3);
            return buf;
        }

        std::string fileNameOf(const std::string& path) { return std::filesystem::u8path(path).filename().u8string(); }

        // The first parameter value of a step that names a file on the cluster ("" none).
        std::string clusterFileOf(const ParamSet& params) {
            const json j = params.toJson();
            for (const auto& [key, value] : j.items()) {
                if (value.is_string() && isRemoteDatasetPath(value.get<std::string>())) return value.get<std::string>();
                if (value.is_array())
                    for (const json& e : value)
                        if (e.is_string() && isRemoteDatasetPath(e.get<std::string>())) return e.get<std::string>();
            }
            return {};
        }

        // Every string of `v` equal to a key of `paths` becomes its value.
        json substituted(const json& v, const std::map<std::string, std::string>& paths) {
            if (v.is_string()) {
                const auto it = paths.find(v.get<std::string>());
                return it == paths.end() ? v : json(it->second);
            }
            if (v.is_array()) {
                json a = json::array();
                for (const json& e : v) a.push_back(substituted(e, paths));
                return a;
            }
            return v;
        }
        json pipelineWithPaths(json p, const std::map<std::string, std::string>& paths) {
            if (paths.empty() || !p.contains("steps") || !p["steps"].is_array()) return p;
            for (json& s : p["steps"])
                if (s.is_object() && s.contains("params") && s["params"].is_object())
                    for (auto& [key, value] : s["params"].items()) value = substituted(value, paths);
            return p;
        }
        // The uploads that still describe the files as they are: path -> node path.
        std::map<std::string, std::string> currentUploads(const std::map<std::string, std::string>& byStamp) {
            std::map<std::string, std::string> out;
            for (const auto& [key, node] : byStamp) {
                const std::size_t nl = key.find('\n');
                if (nl == std::string::npos) continue;
                const std::string path = key.substr(0, nl);
                if (fileStamp(path) == key.substr(nl + 1)) out[path] = node;
            }
            return out;
        }
    } // namespace

    // --- RemoteConfig ------------------------------------------------------------------

    std::unique_ptr<RemoteWorker> RemoteConfig::open(const std::function<bool()>& cancelled) const {
        if (connect) return connect(cancelled);
        return RemoteWorker::connect(host, port, token, std::chrono::seconds(10), cancelled, socksPort);
    }

    // --- the engine on the node, as the workbench talks to it ---------------------------------

    struct Workbench::EngineLink {
        struct Endpoint {
            std::mutex m;
            RemoteConfig config;
            std::unique_ptr<RemoteWorker> open(const std::function<bool()>& cancelled = {}) {
                RemoteConfig c;
                {
                    const std::lock_guard<std::mutex> g(m);
                    c = config;
                }
                return c.open(cancelled);
            }
        };
        struct Answer {
            json result;
            std::vector<rpc::Tensor> tensors;
            std::string error;
        };

        std::shared_ptr<Endpoint> endpoint = std::make_shared<Endpoint>();
        // The node's outputs are drawn through this one (two connections).
        std::shared_ptr<RemoteDatasets> datasets;

        std::mutex qm;
        std::condition_variable qcv;
        std::map<std::string, Answer> answers;                         // by method + params
        std::deque<std::tuple<std::string, std::string, json>> queue;  // key, method, params
        std::set<std::string> asked;
        std::atomic<bool> quit{false};
        std::function<void()> wake;
        std::atomic<bool> arrived{false};
        std::thread thread;

        EngineLink() {
            auto ep = endpoint;
            datasets = std::make_shared<RemoteDatasets>("the cluster node", [ep] { return ep->open(); });
        }
        ~EngineLink() {
            {
                const std::lock_guard<std::mutex> g(qm);
                quit.store(true);
            }
            qcv.notify_all();
            if (thread.joinable()) thread.join();
        }

        void setConfig(const RemoteConfig& c) {
            {
                const std::lock_guard<std::mutex> g(endpoint->m);
                endpoint->config = c;
            }
            const std::lock_guard<std::mutex> g(qm);
            answers.clear();
            asked.clear();
            queue.clear();
        }

        std::optional<Answer> ask(const std::string& method, const json& params) {
            const std::string key = method + "\n" + params.dump();
            const std::lock_guard<std::mutex> g(qm);
            if (auto it = answers.find(key); it != answers.end()) return it->second;
            if (asked.insert(key).second) {
                queue.emplace_back(key, method, params);
                if (!thread.joinable()) thread = std::thread([this] { loop(); });
                qcv.notify_all();
            }
            return std::nullopt;
        }

        void loop() {
            std::unique_ptr<RemoteWorker> w;
            for (;;) {
                std::string key, method;
                json params;
                {
                    std::unique_lock<std::mutex> lk(qm);
                    qcv.wait(lk, [this] { return quit.load() || !queue.empty(); });
                    if (quit.load()) return;
                    std::tie(key, method, params) = std::move(queue.front());
                    queue.pop_front();
                }
                Answer a;
                try {
                    if (!w || !w->isOpen()) w = endpoint->open([this] { return quit.load(); });
                    WorkerResult r = w->call(method, params, {}, {}, [this] { return quit.load(); });
                    a.result = std::move(r.result);
                    a.tensors = std::move(r.tensors);
                } catch (const std::exception& e) {
                    a.error = e.what();
                    if (a.error.rfind("worker: ", 0) == 0) a.error = a.error.substr(8);
                    if (w && !w->isOpen()) w.reset();
                }
                std::function<void()> wakeUp;
                {
                    const std::lock_guard<std::mutex> g(qm);
                    if (!asked.count(key)) continue;   // asked of an engine that was replaced since
                    if (answers.size() > 256) answers.clear();
                    answers[key] = std::move(a);
                    wakeUp = wake;
                }
                arrived.store(true);
                if (wakeUp) wakeUp();
            }
        }
    };

    std::optional<ViewMode> viewModeFromString(const std::string& s) noexcept {
        const std::string l = lower(s);
        if (l == "ortho" || l == "2d") return ViewMode::Ortho;
        if (l == "3d" || l == "volume") return ViewMode::Volume;
        if (l == "compare") return ViewMode::Compare;
        return std::nullopt;
    }
    std::optional<ViewerTool> viewerToolFromString(const std::string& s) noexcept {
        const std::string l = lower(s);
        if (l == "nav" || l == "navigate") return ViewerTool::Navigate;
        if (l == "probe") return ViewerTool::Probe;
        if (l == "measure") return ViewerTool::Measure;
        if (l == "roi") return ViewerTool::Roi;
        if (l == "paint") return ViewerTool::Paint;
        if (l == "prompt") return ViewerTool::Prompt;
        return std::nullopt;
    }
    std::optional<PromptMode> promptModeFromString(const std::string& s) noexcept {
        const std::string l = lower(s);
        for (PromptMode m : {PromptMode::Box, PromptMode::Click, PromptMode::Scribble})
            if (l == toString(m)) return m;
        return std::nullopt;
    }
    std::optional<PaintTool> paintToolFromString(const std::string& s) noexcept {
        static const PaintTool all[] = {PaintTool::Brush, PaintTool::Erase, PaintTool::Fill, PaintTool::Pick,
                                        PaintTool::Merge, PaintTool::Split, PaintTool::Delete, PaintTool::Lasso};
        const std::string l = lower(s);
        for (PaintTool t : all)
            if (l == toString(t)) return t;
        return std::nullopt;
    }

    // --- ViewState ---------------------------------------------------------------

    json ViewState::toJson() const {
        return {{"mode", toString(mode)},
                {"tool", toString(tool)},
                {"paint_tool", toString(paintTool)},
                {"prompt_mode", toString(promptMode)},
                {"brush_px", brushPx},
                {"paint_3d", paint3d},
                {"z", z},
                {"t", t},
                {"crosshair_x", cx},
                {"crosshair_y", cy},
                {"crosshair", crosshair},
                {"labels", labels},
                {"bounding_box", boundingBox},
                {"scale_bar", scaleBar},
                {"physical_z", physicalZ},
                {"sync_zt", syncZT},
                {"channels", channelVisible},
                {"zoom", zoom},
                {"pan", {panX, panY}},
                {"yaw", yaw},
                {"pitch", pitch},
                {"clip_z", {clipZ[0], clipZ[1]}},
                {"label_opacity", labelOpacity},
                {"selected_label", selectedLabel},
                {"solo_label", soloLabel},
                {"trajectories", trajectories},
                {"follow_track", followTrack}};
    }

    ViewState ViewState::fromJson(const json& j) { return fromJson(j, ViewState{}); }

    ViewState ViewState::fromJson(const json& j, const ViewState& base) {
        ViewState s = base;
        if (!j.is_object()) return s;
        auto str = [&](const char* k) { return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : std::string(); };
        if (auto m = viewModeFromString(str("mode"))) s.mode = *m;
        if (auto t = viewerToolFromString(str("tool"))) s.tool = *t;
        if (auto t = paintToolFromString(str("paint_tool"))) s.paintTool = *t;
        if (auto m = promptModeFromString(str("prompt_mode"))) s.promptMode = *m;
        auto num = [&](const char* k, auto& out) {
            if (j.contains(k) && j[k].is_number()) out = static_cast<std::decay_t<decltype(out)>>(j[k].get<double>());
        };
        auto boolean = [&](const char* k, bool& out) {
            if (j.contains(k) && j[k].is_boolean()) out = j[k].get<bool>();
        };
        num("brush_px", s.brushPx);
        boolean("paint_3d", s.paint3d);
        num("z", s.z);
        num("t", s.t);
        num("crosshair_x", s.cx);
        num("crosshair_y", s.cy);
        boolean("crosshair", s.crosshair);
        boolean("labels", s.labels);
        boolean("bounding_box", s.boundingBox);
        boolean("scale_bar", s.scaleBar);
        boolean("physical_z", s.physicalZ);
        boolean("sync_zt", s.syncZT);
        if (j.contains("channels") && j["channels"].is_array()) {
            s.channelVisible.clear();
            for (const json& e : j["channels"]) s.channelVisible.push_back(e.is_boolean() ? e.get<bool>() : true);
        }
        num("zoom", s.zoom);
        if (j.contains("pan") && j["pan"].is_array() && j["pan"].size() == 2) {
            s.panX = j["pan"][0].get<double>();
            s.panY = j["pan"][1].get<double>();
        }
        num("yaw", s.yaw);
        num("pitch", s.pitch);
        if (j.contains("clip_z") && j["clip_z"].is_array() && j["clip_z"].size() == 2)
            s.clipZ = {j["clip_z"][0].get<double>(), j["clip_z"][1].get<double>()};
        num("label_opacity", s.labelOpacity);
        num("selected_label", s.selectedLabel);
        boolean("solo_label", s.soloLabel);
        boolean("trajectories", s.trajectories);
        boolean("follow_track", s.followTrack);
        return s;
    }

    // --- RunJob ------------------------------------------------------------------

    void RunJob::requireFinished(const char* what) const {
        if (!finished()) throw std::logic_error(std::string("RunJob::") + what + " read before the job finished");
    }

    void RunJob::connectWorker() {
        if (backend_ == Backend::Hpc) {
            progress_.set(0.0, -1, "Connecting to the HPC worker…");
            ownedRemote_ = remoteConfig_.open([this] { return cancelled_.load(); });
            if (!ownedRemote_) throw std::runtime_error("no connection to the HPC worker");
            const WorkerCapabilities& caps = ownedRemote_->capabilities();
            onEngine_ = caps.engine.is_object();
            workerNote_ = (onEngine_ ? "HPC engine: " : "HPC worker: ") + caps.device + " on " + caps.hostname;
        } else if (needsWorker_) {
            if (!launcher_) throw std::runtime_error("no Python worker launcher configured");
            progress_.set(0.0, -1, "Starting the Python worker…");
            ownedRemote_ = launcher_();
            if (!ownedRemote_) throw std::runtime_error("the Python worker did not start");
            workerNote_ = "Local worker: " + ownedRemote_->capabilities().device;
        }
        ctx_.remote = ownedRemote_.get();
    }

    void RunJob::execute() {
        const auto t0 = std::chrono::steady_clock::now();
        if (finished()) return;   // runs once
        error_.clear();
        cancelledResult_ = false;
        reports_.clear();
        // The worker first: a process start or a remote handshake can take a
        // while, and that belongs on this thread, not the GUI's. A job
        // cancelled before it got this far (the window closing) does not
        // start a worker it will never use.
        try {
            if (!cancelled_.load()) connectWorker();
        } catch (const WorkerStartError& e) {
            // Kept whole: the host can offer what would fix it (a Python
            // environment to set up, another interpreter), and its hint is
            // that host's own next step.
            workerFailure_ = e;
            error_ = "Worker unavailable: " + withHint(e.what(), e.hint.empty() ? workerHint_ : e.hint);
        } catch (const std::exception& e) {
            // A start that gave up because the run was cancelled (a launcher
            // may stop waiting for the worker then) is a cancellation, not
            // an unavailable worker.
            if (isCancellation(e)) cancelledResult_ = true;
            else error_ = "Worker unavailable: " + withHint(e.what(), workerHint_);
        }
        // The HPC job's engine must be one whose operations are this
        // application's; a job without one runs Python steps only.
        if (error_.empty() && !cancelledResult_ && backend_ == Backend::Hpc && ownedRemote_) {
            if (onEngine_) {
                if (const std::string m = engineMismatch(buildInfo(), buildInfoFromJson(ownedRemote_->capabilities().engine)); !m.empty()) error_ = m;
            } else {
                for (int i = 1; i <= target_; ++i) {
                    const Step& s = pipeline_.at(i);
                    if (!s.enabled || executor_->isFresh(pipeline_, i) || s.op().needsWorker(s.params)) continue;
                    error_ = noEngineRefusal(i, s.name);
                    break;
                }
            }
        }
        if (error_.empty() && !cancelledResult_ && backend_ == Backend::Hpc && onEngine_) {
            try {
                executeOnEngine();
            } catch (const CancelledError&) {
                cancelledResult_ = true;
            } catch (const std::exception& e) {
                if (isCancellation(e)) cancelledResult_ = true;
                else error_ = e.what();
            }
        } else if (error_.empty() && !cancelledResult_) {
            // Whole volumes of cluster data come here only for a run the user
            // chose to compute on this computer; on the HPC backend (the Python
            // worker's job) a step that would download one is refused.
            std::optional<RemoteDownloads::Allow> allow;
            if (backend_ != Backend::Hpc) allow.emplace("a run on this computer");
            // Steps that will actually run, for an overall progress fraction.
            int toRun = 0;
            for (int i = 0; i <= target_; ++i) {
                const Step& s = pipeline_.at(i);
                if ((i == 0 || s.enabled) && !executor_->isFresh(pipeline_, i)) ++toRun;
            }
            int done = 0;
            int currentStep = -1;
            StepContext ctx = ctx_;
            ctx.progress = [&](double f, const std::string& m) {
                const double overall = toRun > 0 ? (done + std::clamp(f, 0.0, 1.0)) / toRun : 1.0;
                progress_.set(overall, currentStep, m);
            };
            ctx.cancelled = [this] { return cancelled_.load(); };
            auto onStep = [&](const StepReport& r) {
                if (r.state == StepReport::State::Running) {
                    currentStep = r.index;
                    progress_.set(toRun > 0 ? static_cast<double>(done) / toRun : 0.0, r.index, "");
                } else if (r.state == StepReport::State::Ran) {
                    ++done;
                    progress_.set(toRun > 0 ? static_cast<double>(done) / toRun : 1.0, r.index, "");
                }
            };
            try {
                output_ = executor_->run(pipeline_, target_, ctx, &reports_, onStep);
            } catch (const CancelledError&) {
                cancelledResult_ = true;
            } catch (const std::exception& e) {
                // isCancellation also accepts the library's untyped
                // "cancelled" (see app/core/cancel.hpp); everything else is a
                // genuine failure and keeps its own message.
                if (isCancellation(e)) cancelledResult_ = true;
                else error_ = e.what();
            } catch (...) {
                error_ = "unknown error";
            }
        }
        if (cancelled_.load()) cancelledResult_ = true;
        if (cancelledResult_) {
            // the run ended because it was asked to, whatever else went wrong
            error_ = "cancelled";
            workerFailure_.reset();
        }
        seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        progress_.set(1.0, -1, "");
        // The worker serves one client at a time: the connection is closed
        // here, on the run's thread, so a client waiting for the worker (the
        // model hub's) does not depend on the GUI thread getting to finishRun.
        ctx_.remote = nullptr;
        ownedRemote_.reset();
        finished_.store(true, std::memory_order_release);
    }

    void RunJob::upload(const UploadFile& f, std::map<std::string, std::string>& nodePaths) {
        const std::string key = f.path + "\n" + f.stamp;
        if (nodePaths.count(key)) return;
        const std::string id = stableHash(key);
        const std::string name = fileNameOf(f.path);
        const std::function<bool()> cancelled = [this] { return cancelled_.load(); };
        const WorkerResult st = ownedRemote_->call("stat_file", {{"key", id}, {"name", name}, {"size", f.bytes}}, {}, {}, cancelled);
        if (st.result.value("exists", false)) {
            nodePaths[key] = st.result.value("path", std::string());
            return;
        }
        std::ifstream in(std::filesystem::u8path(f.path), std::ios::binary);
        if (!in) throw std::runtime_error("cannot read " + f.path + " to upload it");
        crypto::Sha256 hash;
        std::vector<char> chunk(std::size_t{8} << 20);
        std::uint64_t offset = 0;
        std::string nodePath;
        do {
            in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            const std::size_t n = static_cast<std::size_t>(in.gcount());
            if (n == 0 && offset < f.bytes) throw std::runtime_error(f.path + " changed while it was uploaded: run again");
            hash.update(chunk.data(), n);
            json p = {{"key", id}, {"name", name}, {"size", f.bytes}, {"offset", offset}};
            if (offset + n >= f.bytes) p["sha256"] = crypto::toHex(hash.finish());
            const WorkerResult r =
                ownedRemote_->call("put_file", p, {rpc::TensorRef{"data", "uint8", {static_cast<Index>(n)}, chunk.data(), n}}, {}, cancelled);
            offset += n;
            const double f01 = f.bytes > 0 ? static_cast<double>(offset) / static_cast<double>(f.bytes) : 1.0;
            progress_.set(0.0, -1, "Uploading " + name + " \xC2\xB7 " + bytesText(offset) + " of " + bytesText(f.bytes) + " (" + std::to_string(static_cast<int>(f01 * 100.0)) + " %)");
            if (r.result.contains("path")) nodePath = r.result["path"].get<std::string>();
        } while (offset < f.bytes);
        if (nodePath.empty()) throw std::runtime_error("the node did not confirm the upload of " + name);
        nodePaths[key] = nodePath;
    }

    void RunJob::executeOnEngine() {
        const std::function<bool()> cancelled = [this] { return cancelled_.load(); };
        StepContext ctx = ctx_;
        ctx.cancelled = cancelled;
        // The Load step stays this computer's: a cluster dataset's lazy
        // source, or the local file it opened (cheap either way).
        if (!executor_->isFresh(pipeline_, 0)) {
            progress_.set(0.0, 0, "Opening the dataset…");
            executor_->run(pipeline_, 0, ctx, &reports_);
            reports_.clear();
        }
        // Nothing to do when the target is fresh here already.
        if (std::shared_ptr<const StepOutput> fresh = executor_->cached(pipeline_, target_)) {
            for (int i = 0; i <= target_; ++i) {
                StepReport r;
                r.id = pipeline_.at(i).id;
                r.index = i;
                r.state = i > 0 && !pipeline_.at(i).enabled ? StepReport::State::Skipped : StepReport::State::Cached;
                reports_.push_back(r);
            }
            output_ = fresh;
            return;
        }
        // The files of this computer the user agreed to send, then the pipeline
        // as the node knows it: those files under their node paths.
        for (const UploadFile& f : uploads_) {
            if (cancelled_.load()) throw CancelledError();
            upload(f, nodePaths_);
        }
        const json pipelineJson = pipelineWithPaths(pipeline_.toJson(), currentUploads(nodePaths_));
        // What this side holds fresh already: the node leaves out their diagnostics.
        json have = json::array();
        for (int i = 1; i <= target_; ++i)
            if (std::shared_ptr<const StepOutput> out = executor_->cached(pipeline_, i))
                if (const auto* node = dynamic_cast<const NodeOutputSource*>(out->source.get())) have.push_back(node->handle());
        const json params = {{"pipeline", pipelineJson},
                             {"target", target_},
                             {"device", ctx.hpcDevice == HpcDevice::Cpu ? "cpu" : "cuda"},
                             {"hub_token", ctx.hubToken},
                             {"have", have}};
        progress_.set(0.0, -1, "Running on the cluster node…");
        const WorkerResult r = ownedRemote_->callWithFrames(
            "pipeline_run", params, {},
            [this](const json& frame) {
                const int step = frame.value("step", -1);
                progress_.set(frame.value("fraction", 0.0), step, frame.value("message", std::string()));
            },
            cancelled);
        const json& result = r.result;
        engineSession_ = result.value("session", std::string());
        const WorkerCapabilities& caps = ownedRemote_->capabilities();
        std::string where = remoteConfig_.where;
        if (where.empty()) {
            const std::string job = caps.engine.contains("job") && caps.engine["job"].is_object() ? caps.engine["job"].value("id", std::string()) : std::string();
            where = caps.hostname + (job.empty() ? std::string() : " \xC2\xB7 job " + job);
        }
        where_ = where;
        reports_.clear();
        if (result.contains("reports") && result["reports"].is_array())
            for (const json& j : result["reports"]) reports_.push_back(stepReportFromJson(j));
        // The node's outputs, seeded here as if they had just run: handles,
        // drawn at screen size, never downloaded.
        if (result.contains("outputs") && result["outputs"].is_array())
            for (const json& o : result["outputs"]) {
                const int index = o.value("index", -1);
                if (index < 1 || index > target_ || pipeline_.at(index).id != o.value("step_id", StepId{0})) continue;
                const std::string handle = o.value("handle", std::string());
                if (std::shared_ptr<const StepOutput> mine = executor_->cached(pipeline_, index))
                    if (const auto* node = dynamic_cast<const NodeOutputSource*>(mine->source.get()); node && node->handle() == handle) continue;
                auto out = std::make_shared<StepOutput>();
                out->meta = datasetMetaFromJson(o.value("meta", json::object()));
                if (o.value("held", false)) out->source = std::make_shared<NodeOutputSource>(nodeDatasets_, handle, out->meta, where);
                if (o.contains("diagnostics") && o["diagnostics"].is_object()) out->diagnostics = decodeDiagnostics(o["diagnostics"], r.tensors);
                out->note = o.value("note", std::string());
                out->seconds = o.value("seconds", 0.0);
                const json ranOn = o.value("ran_on", json::object());
                out->ranOn = backendFromString(ranOn.value("backend", std::string("CPU"))).value_or(Backend::Cpu);
                out->ranOnDevice = ranOn.value("device", std::string());
                out->where = where;
                if (o.contains("labels") && o["labels"].value("present", false))
                    out->note += std::string(out->note.empty() ? "" : " \xC2\xB7 ") + "labels kept on the node";
                executor_->seed(pipeline_, index, out);
            }
        if (result.contains("error")) throw std::runtime_error(result["error"].get<std::string>());
        output_ = executor_->lastOutput(pipeline_.at(target_).id);
    }

    // --- Workbench ---------------------------------------------------------------

    Workbench::Workbench(std::filesystem::path scratchDir) : executor_(std::move(scratchDir)), engine_(std::make_shared<EngineLink>()) {
        registerBuiltinOperations();
        pipeline_ = Pipeline();
        // Default pipeline of the design: Load + Contrast, Contrast selected and viewed.
        if (findOperation("contrast")) pipeline_.add("contrast");
        selected_ = viewed_ = std::min(1, pipeline_.size() - 1);
        if (!cudaAvailable()) backend_ = Backend::Cpu;
    }

    Workbench::~Workbench() = default;

    void Workbench::addObserver(Observer* o) {
        if (o && std::find(observers_.begin(), observers_.end(), o) == observers_.end()) observers_.push_back(o);
    }

    void Workbench::removeObserver(Observer* o) {
        observers_.erase(std::remove(observers_.begin(), observers_.end(), o), observers_.end());
    }

    void Workbench::notify(void (Observer::*fn)()) {
        // copy: an observer may remove itself while being notified
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) (o->*fn)();
    }

    void Workbench::notifyStep(int index) {
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) o->stepChanged(index);
    }

    void Workbench::logLine(const std::string& line) {
        char stamp[16];
        const std::time_t now = std::time(nullptr);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &now);
#else
        localtime_r(&now, &tm);
#endif
        std::strftime(stamp, sizeof stamp, "%H:%M:%S ", &tm);
        log_.push_back(stamp + line);
        if (log_.size() > 5000) log_.erase(log_.begin(), log_.begin() + 1000);
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) o->logged(log_.back());
    }

    // --- snapshots / undo ----------------------------------------------------------

    Workbench::Snapshot Workbench::snapshot() const {
        Snapshot s;
        s.pipeline = pipeline_.toJson();
        s.selected = selected_;
        s.viewed = viewed_;
        s.view = view_;
        return s;
    }

    void Workbench::restore(const Snapshot& s) {
        // Ids keep counting up: a step added after an undo must not get the
        // id of the step the undo removed, whose cached output the executor
        // still holds (and would show as the new step's).
        const StepId next = pipeline_.peekNextId();
        pipeline_ = Pipeline::fromJson(s.pipeline);
        pipeline_.reserveIds(next);
        selected_ = s.selected;
        viewed_ = s.viewed;
        view_ = s.view;
        clampSelection();
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        notify(&Observer::viewedStepChanged);
        notify(&Observer::viewStateChanged);
        notify(&Observer::outputsChanged);
    }

    void Workbench::pushEdit(const std::string& label, const Snapshot& before, const std::string& mergeKey) {
        // A merged group (slider drag) undoes to the "before" of its first
        // edit; the history says whether this edit continues the group, so
        // a group interrupted by any other entry starts afresh.
        Snapshot first = before;
        if (!mergeKey.empty()) {
            if (history_.mergesWith(mergeKey) && mergeFirst_ && mergeFirst_->first == mergeKey) first = mergeFirst_->second;
            mergeFirst_ = std::make_pair(mergeKey, first);
        }
        const Snapshot after = snapshot();
        Command c;
        c.label = label;
        c.mergeKey = mergeKey;
        c.undo = [this, first] { restore(first); };
        c.redo = [this, after] { restore(after); };
        pushCommand(std::move(c));
    }

    void Workbench::pushCommand(Command c) {
        if (c.mergeKey.empty() || !(mergeFirst_ && mergeFirst_->first == c.mergeKey)) mergeFirst_.reset();
        history_.push(std::move(c));
        notify(&Observer::historyChanged);
    }

    void Workbench::undo() {
        if (refuseIfRunning("undo")) return;
        endPaintStroke();
        if (!history_.canUndo()) return;
        const std::string label = history_.undoLabel();
        history_.undo();
        mergeFirst_.reset();
        logLine("Undo: " + label);
        notify(&Observer::historyChanged);
    }

    void Workbench::redo() {
        if (refuseIfRunning("redo")) return;
        endPaintStroke();
        if (!history_.canRedo()) return;
        const std::string label = history_.redoLabel();
        history_.redo();
        mergeFirst_.reset();
        logLine("Redo: " + label);
        notify(&Observer::historyChanged);
    }

    bool Workbench::refuseIfRunning(const char* what) {
        if (!activeRun_) return false;
        logLine(std::string("Cannot ") + what + " while a run is in progress: cancel it or wait for it to finish.");
        return true;
    }

    void Workbench::clampSelection() {
        const int last = std::max(0, pipeline_.size() - 1);
        selected_ = std::clamp(selected_, 0, last);
        viewed_ = std::clamp(viewed_, 0, last);
    }

    // --- dataset -------------------------------------------------------------------

    namespace {
        // `params` holds what the options do not say (the defaults for a
        // dataset opened afresh, a pipeline's own Load step when it names it).
        void applyOpenOptions(ParamSet& params, const std::string& path, const OpenOptions& o, const DatasetMeta& meta) {
            params.set("path", path);
            params.set("read_as", std::string(o.readAll ? "Full load to RAM" : "Lazy (chunk on demand)"));
            params.set("tile", static_cast<std::int64_t>(o.tile));
            if (o.pageOrder) {
                params.set("page_order", o.pageOrder->order);
                params.set("c", static_cast<std::int64_t>(o.pageOrder->c));
                params.set("t", static_cast<std::int64_t>(o.pageOrder->t));
                params.set("z", static_cast<std::int64_t>(o.pageOrder->z));
            }
            if (o.voxelUm) {
                params.set("voxel_x", (*o.voxelUm)[0]);
                params.set("voxel_y", (*o.voxelUm)[1]);
                params.set("voxel_z", (*o.voxelUm)[2]);
            }
            if (o.sim) {
                params.set("sim_ndirs", static_cast<std::int64_t>(o.sim->present ? o.sim->ndirs : 0));
                params.set("sim_nphases", static_cast<std::int64_t>(o.sim->present ? o.sim->nphases : 0));
                params.set("sim_fast", o.sim->present && o.sim->fastSi);
            } else if (meta.sim.present) {
                params.set("sim_ndirs", static_cast<std::int64_t>(meta.sim.ndirs));
                params.set("sim_nphases", static_cast<std::int64_t>(meta.sim.nphases));
            }
            if (meta.lightSheet) params.set("sheet_angle", meta.sheetAngleDeg);
        }
    } // namespace

    OpenOptions Workbench::openOptionsFromLoadParams(const ParamSet& p) { return loadOpenOptions(p); }

    void Workbench::startRecording(const std::string& path) {
        nlohmann::json header{{"application", "sirius"}};
        if (hasDataset()) {
            header["dataset"] = datasetMeta_.sourcePath;
            header["dims"] = datasetMeta_.dims.toString();
        }
        header["pipeline"] = pipeline_.toJson();
        session_.start(path, header);
        logLine("Recording this session to " + path);
        notify(&Observer::historyChanged);
    }

    void Workbench::stopRecording() {
        if (!session_.recording()) return;
        const std::string where = session_.path().string();
        const std::uint64_t lines = session_.lines();
        session_.stop();   // writes the "stopped" line
        logLine("Stopped recording: " + std::to_string(lines) + " events in " + where);
        notify(&Observer::historyChanged);
    }

    void Workbench::recordEvent(const std::string& event, const nlohmann::json& fields) { session_.record(event, fields); }

    void Workbench::openDataset(const std::string& path, const OpenOptions& options) {
        // A dataset opened afresh (the Open dialog, a recent file, a drop)
        // starts from the Load step's defaults: the previous dataset's page
        // order, axes, voxel size or tile describe that dataset, and would be
        // applied to this one the next time the Load step runs.
        const Operation* op = findOperation("load");
        openDatasetAs(path, options, op ? op->defaults() : ParamSet{});
    }

    void Workbench::adoptDataset(OpenResult opened, const std::string& path, const OpenOptions& options) {
        const Operation* op = findOperation("load");
        installOpened(std::move(opened), path, options, op ? op->defaults() : ParamSet{});
    }

    void Workbench::openDatasetAs(const std::string& path, const OpenOptions& options, ParamSet loadParams) {
        if (refuseIfRunning("open a dataset")) throw std::runtime_error("A run is in progress: cancel it or wait before opening a dataset.");
        installOpened(sirius::app::openDataset(path, options), path, options, std::move(loadParams));
    }

    void Workbench::installOpened(OpenResult opened, const std::string& path, const OpenOptions& options, ParamSet loadParams) {
        if (refuseIfRunning("open a dataset")) throw std::runtime_error("A run is in progress: cancel it or wait before opening a dataset.");
        applyOpenOptions(loadParams, path, options, opened.meta);
        if (const Operation* op = findOperation("load")) {
            loadParams.applyDefaults(op->info().params);
            loadParams.coerce(op->info().params);
        }
        pipeline_.at(0).params = std::move(loadParams);
        installDataset(opened.source, opened.meta, "opened " + opened.meta.format);
        logLine("Opened " + path + " · " + datasetMeta_.shapeString() + " · " + opened.metadataSummary);
        if (!opened.fullLoadSkipped.empty()) logLine("Full load skipped: " + opened.fullLoadSkipped + ".");
        session_.record("dataset", {{"path", path},
                                    {"dims", datasetMeta_.dims.toString()},
                                    {"voxel_um", {datasetMeta_.voxelUm[0], datasetMeta_.voxelUm[1], datasetMeta_.voxelUm[2]}},
                                    {"format", datasetMeta_.format}});
        notify(&Observer::datasetChanged);
        notify(&Observer::pipelineChanged);
        notifyStep(0);
        notify(&Observer::viewStateChanged);
        notify(&Observer::outputsChanged);
    }

    void Workbench::setDataset(std::shared_ptr<ArraySource> source) {
        if (refuseIfRunning("set the dataset")) return;
        if (!source) {
            closeDataset();
            return;
        }
        DatasetMeta meta = source->meta();
        pipeline_.at(0).params.set("path", meta.sourcePath);
        installDataset(std::move(source), std::move(meta), {});
        notify(&Observer::datasetChanged);
        notify(&Observer::pipelineChanged);
        notify(&Observer::viewStateChanged);
        notify(&Observer::outputsChanged);
    }

    void Workbench::installDataset(std::shared_ptr<ArraySource> source, DatasetMeta meta, std::string note) {
        // The part open and set share: the source becomes the Load step's
        // output, the caches start over and the view centres on the data.
        endPaintStroke();
        source_ = std::move(source);
        datasetMeta_ = std::move(meta);
        executor_.clear();
        // A new dataset is a new session: the history entries before it
        // would undo pipeline edits under data they were not made on, and
        // the open itself cannot be undone (the previous source is gone).
        history_.clear();
        mergeFirst_.reset();
        notify(&Observer::historyChanged);
        auto out = std::make_shared<StepOutput>();
        out->meta = datasetMeta_;
        out->source = source_;
        if (source_->inMemory()) out->array = source_->readAll();
        out->note = std::move(note);
        loadOutput_ = out;
        loadOutputParams_ = pipeline_.at(0).params;
        seedLoadOutput();
        view_.channelVisible.assign(static_cast<std::size_t>(std::max<Index>(datasetMeta_.dims.c, 1)), true);
        view_.cx = datasetMeta_.dims.x / 2;
        view_.cy = datasetMeta_.dims.y / 2;
        view_.z = datasetMeta_.dims.z / 2;
        view_.t = 0;
    }

    void Workbench::seedLoadOutput() {
        // The opened dataset is the Load step's output only for the parameters
        // it was opened with: after an edit (a tile, a page order, a voxel
        // size) the step has to run again, and storing the old source under
        // the edited parameters served the old data as their fresh result.
        if (source_ && loadOutput_ && pipeline_.at(0).params.toJson() == loadOutputParams_.toJson())
            executor_.seed(pipeline_, 0, loadOutput_);
    }

    void Workbench::closeDataset() {
        if (refuseIfRunning("close the dataset")) return;
        if (!source_) return;
        endPaintStroke();
        source_.reset();
        loadOutput_.reset();
        datasetMeta_ = DatasetMeta{};
        pipeline_.at(0).params.set("path", std::string());
        executor_.clear();
        history_.clear();
        logLine("Closed dataset");
        notify(&Observer::datasetChanged);
        notify(&Observer::pipelineChanged);
        notify(&Observer::outputsChanged);
        notify(&Observer::historyChanged);
    }

    // --- pipeline edits -------------------------------------------------------------

    StepId Workbench::addStep(const std::string& kind, int at, bool seedParams) {
        if (refuseIfRunning("add a step")) return 0;
        const Snapshot before = snapshot();
        const StepId id = pipeline_.add(kind, at);
        const int index = pipeline_.indexOf(id);
        // Let the operation seed its parameters from the data it will see, and
        // only from that: the nearest enabled step above, with an output that
        // is current. An older output further up (the raw data under a SIM
        // step that has not run yet) is not what this step will get, and a
        // window taken from it froze a range the real input does not have.
        // Without a current input the defaults stay, which work it out at run time.
        int actual = -1;
        const std::shared_ptr<const StepOutput> upstream = seedParams ? upstreamOutput(index, &actual) : nullptr;
        int nearest = index - 1;
        while (nearest > 0 && !pipeline_.at(nearest).enabled) --nearest;
        // An input that stays on the cluster is not read here for it: the
        // defaults (an automatic window) are worked out where the step runs.
        const bool remoteInput = upstream && !upstream->array && upstream->source && upstream->source->viewProvider();
        if (upstream && !remoteInput && actual == nearest && outputFresh(actual)) {
            try {
                Step& s = pipeline_.at(index);
                s.params = s.op().initialParams(s.params, upstream->asInput());
            } catch (const std::exception& e) {
                logLine(std::string("Initial parameters: ") + e.what());
            }
        }
        selected_ = viewed_ = index;
        session_.record("step_added", {{"index", index}, {"kind", kind}, {"params", pipeline_.at(index).params.toJson()}});
        onStepSelected(index);
        pushEdit("Add " + pipeline_.at(index).name, before);
        logLine("Added step " + Step::number(index) + " " + pipeline_.at(index).name);
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        notify(&Observer::viewedStepChanged);
        notify(&Observer::viewStateChanged);
        return id;
    }

    void Workbench::removeStep(int index) {
        if (index < 1 || index >= pipeline_.size()) return;
        if (refuseIfRunning("remove a step")) return;
        endPaintStroke();
        const Snapshot before = snapshot();
        const std::string name = pipeline_.at(index).name;
        const StepId id = pipeline_.at(index).id;
        pipeline_.remove(index);
        executor_.invalidate(id);
        if (selected_ >= index) selected_ = std::max(0, selected_ - 1);
        if (viewed_ >= index) viewed_ = std::max(0, viewed_ - 1);
        clampSelection();
        pushEdit("Remove " + name, before);
        logLine("Removed step " + name);
        session_.record("step_removed", {{"index", index}, {"name", name}});
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        notify(&Observer::viewedStepChanged);
        notify(&Observer::outputsChanged);
    }

    bool Workbench::moveStep(int index, int delta) {
        if (refuseIfRunning("move a step")) return false;
        const Snapshot before = snapshot();
        if (!pipeline_.move(index, delta)) return false;
        const int j = index + delta;
        auto remap = [&](int k) { return k == index ? j : (k == j ? index : k); };
        selected_ = remap(selected_);
        viewed_ = remap(viewed_);
        pushEdit(std::string("Move ") + pipeline_.at(j).name + (delta < 0 ? " up" : " down"), before);
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        notify(&Observer::viewedStepChanged);
        notify(&Observer::outputsChanged);
        return true;
    }

    StepId Workbench::duplicateStep(int index) {
        if (index < 1 || index >= pipeline_.size()) return 0;
        if (refuseIfRunning("duplicate a step")) return 0;
        const Snapshot before = snapshot();
        const StepId id = pipeline_.duplicate(index);
        selected_ = pipeline_.indexOf(id);
        // the copy goes in right below the original: a viewed step further
        // down moved one place, and the view stays on it rather than on
        // whatever step took its old index (a Delete would go there)
        const bool viewedMoved = viewed_ > index;
        if (viewedMoved) ++viewed_;
        pushEdit("Duplicate " + pipeline_.at(index).name, before);
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        if (viewedMoved) notify(&Observer::viewedStepChanged);
        return id;
    }

    void Workbench::setStepEnabled(int index, bool on) {
        if (index < 1 || index >= pipeline_.size() || pipeline_.at(index).enabled == on) return;
        if (refuseIfRunning("enable or skip a step")) return;
        const Snapshot before = snapshot();
        pipeline_.setEnabled(index, on);
        pushEdit(std::string(on ? "Enable " : "Skip ") + pipeline_.at(index).name, before);
        notifyStep(index);
        notify(&Observer::pipelineChanged);
        notify(&Observer::outputsChanged);
    }

    bool Workbench::applyPreset(int index, const std::string& presetName) {
        if (index < 0 || index >= pipeline_.size()) return false;
        const Step& step = pipeline_.at(index);
        const ParamPreset* preset = nullptr;
        for (const ParamPreset& p : step.op().info().presets)
            if (p.name == presetName) preset = &p;
        if (preset == nullptr) return false;
        // setStepParams refuses while a run is in progress, and silently: a
        // preset that reported success there would leave the caller with an
        // undo entry for a change that never happened. Going through
        // refuseIfRunning puts the reason in the log, as every other refused
        // edit does.
        if (refuseIfRunning("apply a preset")) return false;
        ParamSet params = step.params;
        for (const auto& [key, value] : preset->values) params.set(key, value);
        // the values are coerced against the specs the way any edit is
        setStepParams(index, params, "Step " + Step::number(index) + " · preset " + preset->name);
        return true;
    }

    void Workbench::setStepParams(int index, const ParamSet& params, const std::string& label,
                                  const std::string& mergeKey) {
        if (index < 0 || index >= pipeline_.size()) return;
        if (refuseIfRunning("edit parameters")) return;
        const Snapshot before = snapshot();
        pipeline_.setParams(index, params);
        // Nothing changed after coercion: compared as values. A set read back
        // from the snapshot's JSON lists its keys sorted, the step's in spec
        // order, so every step with two parameters or more looked edited, and
        // the undo entry of a no-op commit (a field losing focus) wiped the redo.
        if (pipeline_.at(index).params.toJson() == before.pipeline["steps"][static_cast<std::size_t>(index)]["params"])
            return;
        pushEdit(label.empty() ? "Edit " + pipeline_.at(index).name : label, before, mergeKey);
        // the values themselves, not just the label: a reader wants what was
        // set, on which kind of step, not a sentence about it
        session_.record("params", {{"index", index},
                                   {"kind", pipeline_.at(index).kind},
                                   {"label", label},
                                   {"from", before.pipeline["steps"][static_cast<std::size_t>(index)]["params"]},
                                   {"to", pipeline_.at(index).params.toJson()}});
        notifyStep(index);
        notify(&Observer::outputsChanged);
    }

    void Workbench::setStepParam(int index, const std::string& key, const ParamValue& value, const std::string& mergeKey) {
        if (index < 0 || index >= pipeline_.size()) return;
        const Step& s = pipeline_.at(index);
        ParamSet p = s.params;
        const ParamValue* old = p.find(key);
        const std::string oldText = old ? toDisplayString(*old) : std::string("—");
        p.set(key, value);
        std::string label = "Step " + Step::number(index) + " · " + key + " " + oldText + " → " + toDisplayString(value);
        for (const ParamSpec& spec : s.op().info().params)
            if (spec.key == key) label = "Step " + Step::number(index) + " · " + spec.label + " " + oldText + " → " + toDisplayString(value);
        setStepParams(index, p, label, mergeKey.empty() ? std::string() : mergeKey + "#" + std::to_string(s.id) + "#" + key);
    }

    void Workbench::setStepCache(int index, CachePolicy policy) {
        if (index < 0 || index >= pipeline_.size() || pipeline_.at(index).cache == policy) return;
        if (refuseIfRunning("change a cache policy")) return;
        const Snapshot before = snapshot();
        pipeline_.setCache(index, policy);
        pushEdit(std::string("Cache ") + pipeline_.at(index).name + " · " + toString(policy), before);
        notifyStep(index);
    }

    void Workbench::renameStep(int index, const std::string& name) {
        if (index < 0 || index >= pipeline_.size()) return;
        if (refuseIfRunning("rename a step")) return;
        const Snapshot before = snapshot();
        pipeline_.rename(index, name);
        pushEdit("Rename step " + Step::number(index), before);
        notifyStep(index);
        notify(&Observer::pipelineChanged);
    }

    void Workbench::replacePipeline(const Pipeline& p, const std::string& label) {
        if (refuseIfRunning("replace the pipeline")) return;
        endPaintStroke();
        const Snapshot before = snapshot();
        std::vector<Step> steps = p.steps();
        // Keep our Load step (its params describe the open dataset) unless the
        // incoming pipeline names a dataset path of its own.
        const bool keepLoad = !steps.empty() && steps.front().params.getString("path").empty();
        pipeline_.replaceSteps(steps, keepLoad);
        // The old steps' outputs go: the incoming steps may reuse their ids,
        // and output(index) serves a step's last output fresh or not.
        executor_.clear();
        seedLoadOutput();
        selected_ = std::min(1, pipeline_.size() - 1);
        viewed_ = pipeline_.size() - 1;
        clampSelection();
        onStepSelected(selected_);
        pushEdit(label, before);
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        notify(&Observer::viewedStepChanged);
        notify(&Observer::viewStateChanged);
        notify(&Observer::outputsChanged);
    }

    void Workbench::loadPipeline(const std::string& path) {
        if (refuseIfRunning("load a pipeline")) throw std::runtime_error("A run is in progress: cancel it or wait before loading a pipeline.");
        Pipeline p = Pipeline::load(path);
        // Relative Path parameters (OTF, parameter file, model, flat field)
        // are relative to the pipeline file, so pipelines travel with their data.
        const std::filesystem::path base = std::filesystem::absolute(std::filesystem::path(path)).parent_path();
        for (int i = 0; i < p.size(); ++i) {
            Step& s = p.at(i);
            const Operation* op = findOperation(s.kind);
            if (!op) continue;
            for (const ParamSpec& spec : op->info().params) {
                if (spec.type != ParamType::Path) continue;
                const std::string v = s.params.getString(spec.key);
                if (v.empty() || isRemoteDatasetPath(v) || std::filesystem::path(v).is_absolute()) continue;   // a cluster path is no file here
                std::error_code ec;
                const std::filesystem::path beside = base / v;
                // beside the pipeline file, unless it only exists relative to
                // the working directory; a missing file is named where the
                // pipeline expects it, so validation can say so. With forward
                // slashes, as every other path the application reports.
                if (std::filesystem::exists(beside, ec) || !std::filesystem::exists(std::filesystem::path(v), ec))
                    s.params.set(spec.key, beside.lexically_normal().generic_string());
            }
        }
        const ParamSet loadBefore = pipeline_.at(0).params;
        replacePipeline(p, "Load pipeline " + std::filesystem::path(path).filename().string());
        pipelinePath_ = path;
        logLine("Loaded pipeline " + path);
        for (int i = 1; i < pipeline_.size(); ++i)
            if (pipeline_.at(i).op().info().missing)
                logLine("Step " + Step::number(i) + " '" + pipeline_.at(i).kind + "' is not loaded: reload plugins once its plugin is installed");
        // A pipeline that names its dataset opens it (relative paths resolve
        // against the pipeline file, then the working directory).
        const std::string dataset = pipeline_.at(0).params.getString("path");
        if (dataset.empty()) return;
        // The open dataset is the pipeline's when it is the same file opened
        // the same way; the same file with another tile, page order or voxel
        // size is opened again, the way the pipeline says.
        if (source_ && datasetMeta_.sourcePath == dataset && pipeline_.at(0).params.toJson() == loadOutputParams_.toJson()) return;
        std::filesystem::path resolved = dataset;
        if (resolved.is_relative() && !isRemoteDatasetPath(dataset)) {
            const std::filesystem::path beside = std::filesystem::path(path).parent_path() / resolved;
            std::error_code ec;
            if (std::filesystem::exists(beside, ec)) resolved = beside;
        }
        try {
            // the pipeline's own Load parameters, not the defaults: its light-sheet
            // angle is not an open option, and must not be lost to the open
            const ParamSet wanted = pipeline_.at(0).params;
            openDatasetAs(resolved.generic_string(), openOptionsFromLoadParams(wanted), wanted);
        } catch (const std::exception& e) {
            logLine("The pipeline's dataset could not be opened: " + std::string(e.what()));
            // The data on screen is still the previous dataset: the Load
            // step says so again, instead of naming a file that is not open
            // while its output (the previous data) is served as fresh.
            if (source_) {
                pipeline_.at(0).params = loadBefore;
                seedLoadOutput();
                notifyStep(0);
                notify(&Observer::outputsChanged);
            }
        }
    }

    void Workbench::savePipeline(const std::string& path) const {
        pipeline_.save(path);
        const_cast<Workbench*>(this)->pipelinePath_ = path;
        const_cast<Workbench*>(this)->logLine("Saved pipeline " + path);
    }

    void Workbench::loadExamplePipeline() {
        if (refuseIfRunning("load the example pipeline")) return;
        replacePipeline(Pipeline::example(), "Load example pipeline");
        logLine("Loaded the example pipeline");
    }

    void Workbench::copyParameters(int index) {
        if (index < 0 || index >= pipeline_.size()) return;
        clipboard_ = std::make_pair(pipeline_.at(index).kind, pipeline_.at(index).params);
    }

    bool Workbench::pasteParameters(int index) {
        if (!clipboard_ || index < 0 || index >= pipeline_.size()) return false;
        if (pipeline_.at(index).kind != clipboard_->first) return false;
        if (refuseIfRunning("paste parameters")) return false;
        setStepParams(index, clipboard_->second, "Paste parameters into " + pipeline_.at(index).name);
        return true;
    }

    // --- descriptions -------------------------------------------------------------

    DatasetMeta Workbench::inputMetaOf(int index) const {
        DatasetMeta meta = datasetMeta_;
        for (int i = 0; i < std::min(index, pipeline_.size()); ++i) {
            const Step& s = pipeline_.at(i);
            if (i > 0 && !s.enabled) continue;
            if (const Operation* op = findOperation(s.kind)) {
                try {
                    meta = op->outputMeta(s.params, meta);
                } catch (const std::exception&) {
                }
            }
        }
        return meta;
    }

    DatasetMeta Workbench::outputMetaOf(int index) const {
        if (index < 0 || index >= pipeline_.size()) return datasetMeta_;
        const DatasetMeta in = inputMetaOf(index);
        const Step& s = pipeline_.at(index);
        if (index > 0 && !s.enabled) return in;
        if (const Operation* op = findOperation(s.kind)) {
            try {
                return op->outputMeta(s.params, in);
            } catch (const std::exception&) {
            }
        }
        return in;
    }

    std::string Workbench::stepSummary(int index) const {
        if (index < 0 || index >= pipeline_.size()) return {};
        const Step& s = pipeline_.at(index);
        if (const Operation* op = findOperation(s.kind)) {
            try {
                return op->summary(s.params, inputMetaOf(index));
            } catch (const std::exception& e) {
                return e.what();
            }
        }
        return "unknown operation";
    }

    Validation Workbench::stepValidation(int index) const {
        Validation v;
        if (index < 0 || index >= pipeline_.size()) {
            v.errors.push_back("no such step");
            return v;
        }
        const Step& s = pipeline_.at(index);
        if (index > 0 && !source_) {
            v.errors.push_back("No dataset loaded.");
            return v;
        }
        if (const Operation* op = findOperation(s.kind)) {
            // A file the step reads on the cluster (cluster://...) is checked
            // there, by the engine, which reads it; here there is no such file.
            if (const std::string file = index > 0 ? clusterFileOf(s.params) : std::string(); !file.empty()) {
                if (!remote_.hasEngine()) {
                    v.errors.push_back(fileNameOf(file) + " is on the cluster: SIRIUS's engine there reads it. Connect to the cluster (with the engine) to "
                                                          "use it, or choose a file on this computer.");
                    return v;
                }
                std::string error;
                const std::optional<json> a = askEngine("step_validate", {{"pipeline", nodePipelineJson()}, {"index", index}}, &error);
                if (!a) {
                    v.warnings.push_back("Checking " + fileNameOf(file) + " on the cluster node\xE2\x80\xA6");
                    return v;
                }
                if (!error.empty()) {
                    v.errors.push_back("The cluster node could not check this step: " + error);
                    return v;
                }
                for (const json& e : a->value("errors", json::array()))
                    if (e.is_string()) v.errors.push_back(e.get<std::string>());
                for (const json& w : a->value("warnings", json::array()))
                    if (w.is_string()) v.warnings.push_back(w.get<std::string>());
                return v;
            }
            try {
                return op->validate(s.params, inputMetaOf(index));
            } catch (const std::exception& e) {
                v.errors.push_back(e.what());
            }
        } else {
            v.errors.push_back("unknown operation " + s.kind);
        }
        return v;
    }

    std::size_t Workbench::estimatedBytesOf(int index) const {
        if (index < 0 || index >= pipeline_.size()) return 0;
        const Step& s = pipeline_.at(index);
        if (const Operation* op = findOperation(s.kind)) {
            try {
                return op->estimatedOutputBytes(s.params, inputMetaOf(index));
            } catch (const std::exception&) {
            }
        }
        return 0;
    }

    // --- selection & view -------------------------------------------------------------

    void Workbench::onStepSelected(int index) {
        if (index < 0 || index >= pipeline_.size()) return;
        const std::string& kind = pipeline_.at(index).kind;
        const OpInfo& info = pipeline_.at(index).op().info();
        if (kind == "volrec") view_.mode = ViewMode::Volume;
        // A Prompt step is made by pointing at objects, so selecting one
        // picks the tool that places the points; its masks are labels.
        if (isPromptStep(pipeline_.at(index).params)) {
            view_.labels = true;
            view_.tool = ViewerTool::Prompt;
        } else if (info.producesLabels || info.needsLabels) {
            view_.labels = true;
            view_.tool = ViewerTool::Paint;
        } else if (view_.tool == ViewerTool::Paint || view_.tool == ViewerTool::Prompt) {
            view_.tool = ViewerTool::Probe;
        }
    }

    void Workbench::select(int index) {
        if (index < 0 || index >= pipeline_.size() || index == selected_) return;
        selected_ = index;
        const ViewState before = view_;
        onStepSelected(index);
        notify(&Observer::selectionChanged);
        if (!(before.mode == view_.mode && before.tool == view_.tool && before.labels == view_.labels))
            notify(&Observer::viewStateChanged);
    }

    void Workbench::view(int index) {
        if (index < 0 || index >= pipeline_.size() || index == viewed_) return;
        viewed_ = index;
        const DatasetMeta meta = outputMetaOf(index);
        view_.channelVisible.resize(static_cast<std::size_t>(std::max<Index>(meta.dims.c, 1)), true);
        syncLabelStats();
        notify(&Observer::viewedStepChanged);
    }

    void Workbench::syncLabelStats() {
        if (running()) return;   // the worker may be reading the statistics
        auto labels = viewedLabels();
        if (!labels || labels->empty() || labels->t() <= 1) return;
        if (view_.t < 0 || view_.t >= labels->t() || labels->statsT() == view_.t) return;
        endPaintStroke();
        labels->recomputeStats(view_.t);
        int actual = -1;
        displayOutput(&actual);
        notifyLabels(actual >= 0 ? pipeline_.at(actual).id : 0);
    }

    void Workbench::setViewState(const ViewState& s) {
        const bool jump = s.soloLabel && s.selectedLabel != 0 && s.selectedLabel != view_.selectedLabel;
        const bool tChanged = s.t != view_.t;
        const bool follow = s.followTrack && (tChanged || !view_.followTrack || s.selectedLabel != view_.selectedLabel);
        view_ = s;
        if (tChanged) syncLabelStats();
        // inspecting one label at a time: a new selection brings it into view
        if (jump) centreOnLabel(s.selectedLabel);
        if (follow) followSelectedTrack();
        notify(&Observer::viewStateChanged);
    }

    std::vector<TrackSummary> Workbench::viewedTrackSummaries() const {
        const std::shared_ptr<const StepOutput> out = displayOutput();
        if (!out || !out->labels || !out->labels->tracked()) return {};
        const std::shared_ptr<const TrackIndex> index = out->labels->tracks();
        if (!index) return {};
        const std::array<double, 3>& v = out->meta.voxelUm;   // x, y, z
        return summarizeTracks(*index, out->labels->lineage(), {v[2], v[1], v[0]});
    }

    bool Workbench::followSelectedTrack() {
        const std::shared_ptr<LabelVolume> labels = viewedLabels();
        if (!labels || !labels->tracked() || view_.selectedLabel == 0) return false;
        const std::shared_ptr<const TrackIndex> index = labels->tracks();
        const std::optional<TrackPoint> p = index ? index->pointAt(view_.selectedLabel, view_.t) : std::nullopt;
        if (!p) return false;   // missing from this frame: stay where the eye is
        const auto near = [](double c, Index n) { return std::clamp<Index>(static_cast<Index>(std::floor(c)), 0, std::max<Index>(n - 1, 0)); };
        view_.z = near(p->centroid[0], labels->z());
        view_.cy = near(p->centroid[1], labels->y());
        view_.cx = near(p->centroid[2], labels->x());
        return true;
    }

    bool Workbench::focusTrack(std::uint32_t id) {
        const std::shared_ptr<LabelVolume> labels = viewedLabels();
        const std::shared_ptr<const TrackIndex> index = labels && labels->tracked() ? labels->tracks() : nullptr;
        const std::optional<TrackPoint> p = index && id ? index->nearestPoint(id, view_.t) : std::nullopt;
        if (!p) return false;
        endPaintStroke();
        view_.selectedLabel = id;
        view_.labels = true;
        if (view_.t != p->t) {
            view_.t = p->t;
            syncLabelStats();
        }
        followSelectedTrack();
        notify(&Observer::viewStateChanged);
        return true;
    }

    void Workbench::setFollowTrack(bool on) {
        if (view_.followTrack == on) return;
        view_.followTrack = on;
        if (on) followSelectedTrack();
        notify(&Observer::viewStateChanged);
    }

    bool Workbench::centreOnLabel(std::uint32_t id) {
        endPaintStroke();   // the statistics must include the stroke
        auto labels = viewedLabels();
        const LabelStats* st = labels ? labels->statsOf(id) : nullptr;
        if (!st) return false;
        view_.z = (st->bbox[0] + st->bbox[1]) / 2;
        view_.cy = (st->bbox[2] + st->bbox[3]) / 2;
        view_.cx = (st->bbox[4] + st->bbox[5]) / 2;
        return true;
    }

    void Workbench::focusLabel(std::uint32_t id) {
        view_.selectedLabel = id;
        view_.labels = true;
        centreOnLabel(id);
        notify(&Observer::viewStateChanged);
    }

    void Workbench::toggleSoloLabel() {
        view_.soloLabel = !view_.soloLabel;
        if (view_.soloLabel) {
            view_.labels = true;
            if (view_.selectedLabel) centreOnLabel(view_.selectedLabel);
        }
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setViewMode(ViewMode m) {
        if (view_.mode == m) return;
        view_.mode = m;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setTool(ViewerTool t) {
        if (view_.tool == t) return;
        view_.tool = t;
        if (t == ViewerTool::Paint || t == ViewerTool::Prompt) view_.labels = true;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setPaintTool(PaintTool t) {
        view_.paintTool = t;
        view_.tool = ViewerTool::Paint;
        view_.labels = true;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setZ(Index z) {
        const DatasetMeta meta = displayedMeta();
        z = std::clamp<Index>(z, 0, std::max<Index>(meta.dims.z - 1, 0));
        if (view_.z == z) return;
        view_.z = z;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setT(Index t) {
        const DatasetMeta meta = displayedMeta();
        t = std::clamp<Index>(t, 0, std::max<Index>(meta.dims.t - 1, 0));
        if (view_.t == t) return;
        view_.t = t;
        syncLabelStats();
        if (view_.followTrack) followSelectedTrack();
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setCrosshair(Index x, Index y, Index z) {
        const DatasetMeta meta = displayedMeta();
        view_.cx = std::clamp<Index>(x, 0, std::max<Index>(meta.dims.x - 1, 0));
        view_.cy = std::clamp<Index>(y, 0, std::max<Index>(meta.dims.y - 1, 0));
        view_.z = std::clamp<Index>(z, 0, std::max<Index>(meta.dims.z - 1, 0));
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setChannelVisible(Index c, bool on) {
        if (c < 0) return;
        if (static_cast<std::size_t>(c) >= view_.channelVisible.size()) view_.channelVisible.resize(static_cast<std::size_t>(c) + 1, true);
        view_.channelVisible[static_cast<std::size_t>(c)] = on;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::toggleCrosshair() {
        view_.crosshair = !view_.crosshair;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::toggleLabels() {
        view_.labels = !view_.labels;
        notify(&Observer::viewStateChanged);
    }

    // --- outputs ---------------------------------------------------------------------

    std::shared_ptr<const StepOutput> Workbench::output(int index) const {
        if (index < 0 || index >= pipeline_.size()) return nullptr;
        if (index == 0) return loadOutput_;
        return executor_.lastOutput(pipeline_.at(index).id);
    }

    bool Workbench::outputFresh(int index) const {
        if (index < 0 || index >= pipeline_.size()) return false;
        return executor_.isFresh(pipeline_, index);
    }

    std::shared_ptr<const StepOutput> Workbench::displayOutput(int* actualIndex) const {
        // A live-preview step that has not run (or is stale) is shown on its
        // input: the viewer applies the step's parameters itself.
        if (viewedIsLivePreview()) return upstreamOutput(viewed_, actualIndex);
        for (int i = viewed_; i >= 0; --i) {
            if (i > 0 && !pipeline_.at(i).enabled) continue;
            auto out = output(i);
            if (out && (out->array || out->source)) {
                if (actualIndex) *actualIndex = i;
                return out;
            }
        }
        if (actualIndex) *actualIndex = -1;
        return nullptr;
    }

    DatasetMeta Workbench::displayedMeta() const {
        if (const std::shared_ptr<const StepOutput> out = displayOutput()) return out->meta;
        return outputMetaOf(viewed_);
    }

    std::shared_ptr<const StepOutput> Workbench::upstreamOutput(int index, int* actualIndex) const {
        for (int i = std::min(index, pipeline_.size()) - 1; i >= 0; --i) {
            if (i > 0 && !pipeline_.at(i).enabled) continue;
            auto out = output(i);
            if (out && (out->array || out->source)) {
                if (actualIndex) *actualIndex = i;
                return out;
            }
        }
        if (actualIndex) *actualIndex = -1;
        return nullptr;
    }

    bool Workbench::viewedIsLivePreview() const {
        if (!source_ || viewed_ <= 0 || viewed_ >= pipeline_.size()) return false;
        const Step& s = pipeline_.at(viewed_);
        if (!s.enabled) return false;
        const Operation* op = findOperation(s.kind);
        return op && op->info().livePreview && !outputFresh(viewed_);
    }

    Diagnostics Workbench::previewDiagnostics(int index) const {
        Diagnostics d;
        if (index < 0 || index >= pipeline_.size()) return d;
        const Step& s = pipeline_.at(index);
        const Operation* op = findOperation(s.kind);
        if (!op) return d;
        d.kind = op->info().diagnostics;
        d.summary = stepSummary(index);
        const DatasetMeta in = inputMetaOf(index), out = outputMetaOf(index);
        d.facts.push_back({"Input", index > 0 ? in.shapeString() : std::string("—")});
        d.facts.push_back({"Output", out.shapeString()});
        const std::size_t bytes = estimatedBytesOf(index);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.1f GB", static_cast<double>(bytes) / 1e9);
        d.facts.push_back({"Est. output", bytes ? buf : "—"});
        const Validation v = stepValidation(index);
        for (const std::string& w : v.warnings) d.warnings.push_back(w);
        for (const std::string& e : v.errors) d.warnings.push_back(e);
        // The operation's own live preview, when it has one and an input exists.
        if (source_ && index > 0) {
            std::shared_ptr<const StepOutput> upstream = upstreamOutput(index);
            // An input that stays on the cluster is previewed there, by the
            // engine, on the data as the node holds it: nothing comes here but
            // the diagnostics (answered on the engine's thread; poll()).
            // (an input not computed on the node yet has no preview: none is guessed from another step's data)
            if (inputOnCluster(index) && !previewedOnNode(index)) return d;
            if (previewedOnNode(index)) {
                std::string error;
                std::vector<rpc::Tensor> tensors;
                const std::optional<json> a = askEngine("step_preview", {{"pipeline", nodePipelineJson()}, {"index", index}}, &error, &tensors);
                if (!a) {
                    d.warnings.push_back("Computing the preview on the cluster node\xE2\x80\xA6");
                    return d;
                }
                if (!error.empty()) {
                    d.warnings.push_back("Preview on the cluster node: " + error);
                    return d;
                }
                if (a->contains("diagnostics") && (*a)["diagnostics"].is_object()) {
                    try {
                        Diagnostics p = decodeDiagnostics((*a)["diagnostics"], tensors);
                        p.warnings.insert(p.warnings.end(), d.warnings.begin(), d.warnings.end());
                        if (p.summary.empty()) p.summary = d.summary;
                        return p;
                    } catch (const std::exception& e) {
                        d.warnings.push_back(e.what());
                    }
                }
                return d;
            }
            if (upstream) {
                try {
                    if (auto p = op->preview(upstream->asInput(), s.params)) {
                        p->warnings.insert(p->warnings.end(), d.warnings.begin(), d.warnings.end());
                        if (p->summary.empty()) p->summary = d.summary;
                        return *p;
                    }
                } catch (const std::exception& e) {
                    d.warnings.push_back(e.what());
                }
            }
        }
        return d;
    }

    Diagnostics Workbench::selectedDiagnostics() const { return diagnosticsOf(selected_); }

    Diagnostics Workbench::diagnosticsOf(int index) const {
        if (index < 0 || index >= pipeline_.size()) return {};
        if (auto out = output(index); out && !out->diagnostics.empty() && index > 0) {
            Diagnostics d = out->diagnostics;
            if (!outputFresh(index)) d.warnings.insert(d.warnings.begin(), "Parameters changed since this result: run the step again.");
            return d;
        }
        return previewDiagnostics(index);
    }

    void Workbench::clearCache(int index) {
        if (index < 1 || index >= pipeline_.size()) return;
        if (refuseIfRunning("clear a cache")) return;
        endPaintStroke();
        executor_.invalidate(pipeline_.at(index).id);
        logLine("Cleared cache of " + pipeline_.at(index).name);
        notify(&Observer::outputsChanged);
    }

    void Workbench::clearAllCaches() {
        if (refuseIfRunning("clear the caches")) return;
        endPaintStroke();
        executor_.clear();
        seedLoadOutput();
        logLine("Cleared all caches");
        notify(&Observer::outputsChanged);
    }

    std::size_t Workbench::cachedBytes() const { return executor_.cachedBytes(); }

    // --- running ------------------------------------------------------------------

    void Workbench::setBackend(Backend b) {
        if (backend_ == b) return;
        backend_ = b;
        logLine(std::string("Backend: ") + toString(b));
        notify(&Observer::backendChanged);
    }

    void Workbench::setCudaDevice(int index) {
        if (index < 0) cudaDevice_ = kAllCudaDevices;
        else {
            const int n = cudaDeviceCount();
            cudaDevice_ = n > 0 ? std::min(index, n - 1) : 0;
        }
        notify(&Observer::backendChanged);
    }

    void Workbench::setHpcDevice(HpcDevice d) {
        if (hpcDevice_ == d) return;
        hpcDevice_ = d;
        logLine(std::string("HPC device: ") + toString(d));
        notify(&Observer::backendChanged);
    }

    void Workbench::setRemoteConfig(RemoteConfig c) {
        remote_ = std::move(c);
        engine_->setConfig(remote_);
        notify(&Observer::backendChanged);
    }

    int Workbench::nodeOutputsGone(const std::string& session, const std::string& reason) {
        int n = 0;
        for (int i = 1; i < pipeline_.size(); ++i) {
            const StepId id = pipeline_.at(i).id;
            const std::optional<Executor::Held> h = executor_.held(id);
            if (!h || !h->output) continue;
            auto* node = dynamic_cast<NodeOutputSource*>(h->output->source.get());
            if (!node || (!session.empty() && node->session() != session)) continue;
            node->markGone(reason);
            if (executor_.dropData(id, reason)) {
                ++n;
                logLine("Step " + Step::number(i) + " " + pipeline_.at(i).name + ": its result is gone, " + reason + ". Run it again.");
            }
        }
        if (session.empty() || session == engineSession_) engineSession_.clear();
        if (n > 0) notify(&Observer::outputsChanged);
        return n;
    }

    std::vector<UploadFile> Workbench::filesToUpload(int target) const {
        std::vector<UploadFile> out;
        if (target < 0 || target >= pipeline_.size()) target = pipeline_.size() - 1;
        const auto consider = [&out](const std::string& v) {
            if (v.empty() || isRemoteDatasetPath(v)) return;
            std::error_code ec;
            const std::filesystem::path p = std::filesystem::u8path(v);
            if (!std::filesystem::is_regular_file(p, ec)) return;
            for (const UploadFile& f : out)
                if (f.path == v) return;
            UploadFile f;
            f.path = v;
            f.bytes = static_cast<std::uint64_t>(std::filesystem::file_size(p, ec));
            f.stamp = fileStamp(v);
            out.push_back(std::move(f));
        };
        for (int i = 0; i <= target; ++i) {
            const Step& s = pipeline_.at(i);
            if (i > 0 && !s.enabled) continue;
            if (i == 0) {
                consider(s.params.getString("path"));
                continue;
            }
            const Operation* op = findOperation(s.kind);
            if (!op) continue;
            for (const ParamSpec& spec : op->info().params) {
                if (spec.type == ParamType::Path) consider(s.params.getString(spec.key));
                else if (spec.type == ParamType::StringList)
                    for (const std::string& v : s.params.getStringList(spec.key)) consider(v);
            }
        }
        return out;
    }

    void Workbench::allowUploads(const std::vector<UploadFile>& files) {
        for (const UploadFile& f : files) uploadConsent_.insert(f.path + "\n" + f.stamp);
    }

    std::string Workbench::placementOf(int index) const {
        if (index < 1 || index >= pipeline_.size()) return {};
        const std::optional<Executor::Held> h = executor_.held(pipeline_.at(index).id);
        if (!h || !h->output) return {};
        std::string tag = placementTag(*h->output);
        if (!h->output->gone.empty()) tag += " \xC2\xB7 gone";
        return tag;
    }

    void Workbench::setWakeHandler(std::function<void()> wake) {
        const std::lock_guard<std::mutex> g(engine_->qm);
        engine_->wake = std::move(wake);
    }

    bool Workbench::poll() {
        if (!engine_->arrived.exchange(false)) return false;
        notify(&Observer::outputsChanged);
        return true;
    }

    json Workbench::nodePipelineJson() const { return pipelineWithPaths(pipeline_.toJson(), currentUploads(nodePaths_)); }

    bool Workbench::inputOnCluster(int index) const {
        if (!source_ || index < 1 || index >= pipeline_.size() || !remote_.hasEngine()) return false;
        const std::shared_ptr<const StepOutput> up = upstreamOutput(index);
        return up && !up->array && up->source && up->source->viewProvider();
    }

    bool Workbench::previewedOnNode(int index) const {
        if (!inputOnCluster(index)) return false;
        // the step's own input, computed: the nearest enabled step above it, fresh
        int actual = -1;
        (void)upstreamOutput(index, &actual);
        int nearest = index - 1;
        while (nearest > 0 && !pipeline_.at(nearest).enabled) --nearest;
        return actual == nearest && outputFresh(actual);
    }

    std::optional<Diagnostics> Workbench::nodePreview(int index, const ParamSet& params) const {
        std::string error;
        std::vector<rpc::Tensor> tensors;
        const std::optional<json> a =
            askEngine("step_preview", {{"pipeline", nodePipelineJson()}, {"index", index}, {"params", params.toJson()}}, &error, &tensors);
        if (!a) return std::nullopt;
        if (!error.empty()) throw std::runtime_error(error);
        if (!a->contains("diagnostics") || !(*a)["diagnostics"].is_object()) throw std::runtime_error("the node has no preview of this step");
        return decodeDiagnostics((*a)["diagnostics"], tensors);
    }

    std::optional<ContrastWindow> Workbench::contrastWindowOf(int index, const ParamSet& params, Index c, bool wantRange) const {
        const std::shared_ptr<const StepOutput> up = upstreamOutput(index);
        if (!up) return std::nullopt;
        if (!inputOnCluster(index)) return contrastWindow(up->asInput(), params, c, 8, wantRange);
        if (!previewedOnNode(index)) return std::nullopt;   // its input is not computed on the node yet
        const std::optional<Diagnostics> d = nodePreview(index, params);
        if (!d) return std::nullopt;
        if (c < 0 || static_cast<std::size_t>(c) >= d->histograms.size()) throw std::runtime_error("the node's preview has no channel " + std::to_string(c));
        const DiagnosticHistogram& h = d->histograms[static_cast<std::size_t>(c)];
        ContrastWindow w;
        w.lo = static_cast<float>(h.lo);
        w.hi = static_cast<float>(h.hi);
        w.gamma = static_cast<float>(params.getDouble("gamma", 1.0));
        w.dataMin = static_cast<float>(h.binLo);
        w.dataMax = static_cast<float>(h.binHi);
        return w;
    }

    std::optional<ParamSet> Workbench::contrastAutoOf(int index, const ParamSet& current) const {
        const std::shared_ptr<const StepOutput> up = upstreamOutput(index);
        if (!up) return std::nullopt;
        if (!inputOnCluster(index)) return contrastAutoParams(current, up->asInput());
        if (!previewedOnNode(index)) return std::nullopt;
        // the automatic window is the preview's own when min / max say automatic
        ParamSet p = current;
        if (const Operation* op = findOperation("contrast")) p.applyDefaults(op->info().params);
        ParamSet automatic = p;
        automatic.set("min", 0.0);
        automatic.set("max", 0.0);
        const std::optional<Diagnostics> d = nodePreview(index, automatic);
        if (!d) return std::nullopt;
        float lo = std::numeric_limits<float>::infinity(), hi = -lo;
        for (const DiagnosticHistogram& h : d->histograms) {
            lo = std::min(lo, static_cast<float>(h.lo));
            hi = std::max(hi, static_cast<float>(h.hi));
        }
        if (!(lo < hi)) {
            lo = 0.0f;
            hi = 1.0f;
        }
        p.set("min", static_cast<double>(lo));
        p.set("max", static_cast<double>(hi));
        return p;
    }

    std::optional<ParamSet> Workbench::contrastResetOf(int index, const ParamSet& current) const {
        const std::shared_ptr<const StepOutput> up = upstreamOutput(index);
        if (!up) return std::nullopt;
        if (!inputOnCluster(index)) return contrastResetParams(current, up->asInput());
        if (!previewedOnNode(index)) return std::nullopt;
        const std::optional<Diagnostics> d = nodePreview(index, current);
        if (!d) return std::nullopt;
        float mn = std::numeric_limits<float>::infinity(), mx = -mn;
        for (const DiagnosticHistogram& h : d->histograms) {
            mn = std::min(mn, static_cast<float>(h.binLo));
            mx = std::max(mx, static_cast<float>(h.binHi));
        }
        if (!(mn < mx)) {
            mn = 0.0f;
            mx = 1.0f;
        }
        ParamSet p = current;
        p.set("min", static_cast<double>(mn));
        p.set("max", static_cast<double>(mx));
        p.set("gamma", 1.0);
        return p;
    }

    std::optional<json> Workbench::askEngine(const std::string& method, const json& params, std::string* error,
                                             std::vector<rpc::Tensor>* tensors) const {
        std::optional<EngineLink::Answer> a = engine_->ask(method, params);
        if (!a) return std::nullopt;
        if (error) *error = a->error;
        if (tensors) *tensors = a->tensors;
        return a->result;
    }

    int Workbench::loadPlugins(bool reload) {
        pluginError_.clear();
        pluginWorkerFailure_.reset();
        // A refused attempt says so too: an empty pluginError() would read
        // as a load that reached the worker and found nothing.
        if (refuseIfRunning("load plugins")) {
            pluginError_ = "a run is in progress";
            return 0;
        }
        if (!launcher_) {
            pluginError_ = "no Python worker launcher configured";
            logLine("Plugins: " + withHint(pluginError_, workerHint_));
            return 0;
        }
        try {
            logLine(std::string("Plugins: ") + (reload ? "reloading" : "loading") +
                    " through the Python worker (starting it first when it is not running)…");
            std::unique_ptr<RemoteWorker> worker = launcher_();
            if (!worker) throw std::runtime_error("the Python worker did not start");
            const PluginLoadResult r = registerPluginOperations(*worker, reload);
            plugins_.clear();
            for (const PluginLoadResult::Entry& e : r.entries) plugins_.push_back({e.kind, e.name, e.file, e.error});
            pluginDirs_ = r.dirs;
            for (const std::string& e : r.errors) logLine("Plugin error: " + e);
            std::string kinds;
            for (const std::string& k : r.kinds) kinds += (kinds.empty() ? "" : ", ") + k;
            logLine(r.kinds.empty() ? "Plugins: none found" + (r.dirs.empty() ? std::string() : " in " + r.dirs.back())
                                    : "Plugins: " + kinds);
            for (const std::string& k : r.removed) logLine("Plugins: no file provides '" + k + "' any more");
            // A step of a kind that is loaded now (a pipeline opened before its
            // plugin was) takes the parameters the operation declares; a step
            // whose plugin went is shown as not loaded.
            bool stepsChanged = false;
            for (int i = 1; i < pipeline_.size(); ++i) {
                const std::string& kind = pipeline_.at(i).kind;
                if (std::find(r.kinds.begin(), r.kinds.end(), kind) != r.kinds.end()) {
                    pipeline_.setParams(i, pipeline_.at(i).params);
                    stepsChanged = true;
                } else if (std::find(r.removed.begin(), r.removed.end(), kind) != r.removed.end()) {
                    stepsChanged = true;
                }
            }
            notify(&Observer::operationsChanged);
            if (stepsChanged) {
                notify(&Observer::pipelineChanged);
                notify(&Observer::outputsChanged);
            }
            return static_cast<int>(r.kinds.size());
        } catch (const WorkerStartError& e) {
            // The start failure is kept whole, as a run keeps it: the host
            // offers its fix from there (the window's Python set-up prompt).
            pluginError_ = e.what();
            pluginWorkerFailure_ = e;
            logLine("Plugins unavailable: " + withHint(e.what(), e.hint.empty() ? workerHint_ : e.hint));
            return 0;
        } catch (const std::exception& e) {
            pluginError_ = e.what();
            logLine("Plugins unavailable: " + withHint(e.what(), workerHint_));
            return 0;
        }
    }

    std::shared_ptr<RunJob> Workbench::refuseRun(RunRefusal::Kind kind, int step, const std::string& line) {
        lastRunRefusal_.kind = kind;
        lastRunRefusal_.step = step;
        lastRunRefusal_.message = line;
        logLine(line);
        return nullptr;
    }

    std::string noEngineRefusal(int index, const std::string& stepName) {
        return "Step " + Step::number(index) + " " + stepName +
               " needs SIRIUS's C++ engine on the cluster, and this job has none: open Cluster \xE2\x96\xB8 Job \xE2\x96\xB8 More options, set Engine "
               "builds folder, then Restart worker.";
    }

    RunGate Workbench::runGate() const {
        if (backend_ != Backend::Hpc || remote_.hasEngine()) return {};
        // not known (sirius-cli --hpc host:port): the run finds out when it connects
        if (!remote_.known) return {};
        return {false, remote_.noEngine.empty() ? std::string(kHpcNoEngine) : remote_.noEngine};
    }

    std::shared_ptr<RunJob> Workbench::createRun(int target) {
        lastRunRefusal_ = RunRefusal{};
        if (activeRun_) return refuseRun(RunRefusal::Kind::Running, -1, "A run is already in progress.");
        if (target < 0 || target >= pipeline_.size()) target = pipeline_.size() - 1;
        // The HPC backend runs nothing without SIRIUS's engine: the first
        // built-in step that would run is named, else the gate's reason.
        if (const RunGate gate = runGate(); !gate.enabled) {
            for (int i = 1; i <= target; ++i) {
                const Step& s = pipeline_.at(i);
                if (!s.enabled || executor_.isFresh(pipeline_, i) || s.op().needsWorker(s.params)) continue;
                return refuseRun(RunRefusal::Kind::NoEngine, i, noEngineRefusal(i, s.name));
            }
            return refuseRun(RunRefusal::Kind::NoEngine, -1, gate.why);
        }
        if (!source_) return refuseRun(RunRefusal::Kind::NoDataset, -1, "Open a dataset before running.");
        bool needsWorker = false;
        for (int i = 1; i <= target; ++i) {
            const Step& s = pipeline_.at(i);
            if (!s.enabled) continue;
            const Validation v = stepValidation(i);
            if (!v.ok())
                return refuseRun(RunRefusal::Kind::Invalid, i, "Step " + Step::number(i) + " " + s.name + " cannot run: " + v.firstError());
            if (s.op().needsWorker(s.params) && !executor_.isFresh(pipeline_, i)) needsWorker = true;
        }
        // Refused before anything is prepared, as the refusals above are.
        if (backend_ != Backend::Hpc && needsWorker && !launcher_)
            return refuseRun(RunRefusal::Kind::NoLauncher, -1,
                             "Worker unavailable: " + withHint("no Python worker launcher configured", workerHint_));
        // The HPC backend never computes here in silence: with SIRIUS's engine
        // on the node the whole run goes there (an engine of other operations
        // is refused; files of this computer go only when the user agrees),
        // and a job with the Python worker only runs the Python steps.
        std::vector<UploadFile> uploads;
        if (backend_ == Backend::Hpc && remote_.hasEngine()) {
            if (const std::string m = engineMismatch(buildInfo(), buildInfoFromJson(remote_.engine)); !m.empty())
                return refuseRun(RunRefusal::Kind::EngineMismatch, -1, m);
            const std::string data = pipeline_.at(0).params.getString("path");
            std::error_code ec;
            if (!data.empty() && !isRemoteDatasetPath(data) && std::filesystem::is_directory(std::filesystem::u8path(data), ec))
                return refuseRun(RunRefusal::Kind::NeedsUpload, 0,
                                 "The dataset is a folder on this computer: the HPC backend computes on the cluster node, and a folder is not "
                                 "uploaded. Open the dataset from the cluster (cluster://\xE2\x80\xA6), or choose CPU/CUDA to run here.");
            std::vector<UploadFile> missing;
            std::uint64_t bytes = 0;
            for (UploadFile& f : filesToUpload(target)) {
                if (uploadConsent_.count(f.path + "\n" + f.stamp)) {
                    uploads.push_back(f);
                } else {
                    bytes += f.bytes;
                    missing.push_back(std::move(f));
                }
            }
            if (!missing.empty()) {
                std::string names;
                for (const UploadFile& f : missing) names += (names.empty() ? "" : ", ") + fileNameOf(f.path) + " (" + bytesText(f.bytes) + ")";
                const bool dataset = missing.front().path == data;
                lastRunRefusal_.uploads = missing;
                return refuseRun(RunRefusal::Kind::NeedsUpload, -1,
                                 std::string(dataset ? "The dataset is on this computer: " : "Files of this computer: ") + names +
                                     ". The HPC backend computes on the cluster node, so " + bytesText(bytes) +
                                     " would be uploaded there first. Upload them, or open the data from the cluster (cluster://\xE2\x80\xA6).");
            }
        }
        endPaintStroke();
        auto job = std::make_shared<RunJob>();
        job->pipeline_ = pipeline_;
        job->target_ = target;
        job->executor_ = &executor_;
        job->ctx_.backend = backend_;
        if (backend_ == Backend::Cuda && cudaAvailable()) {
            job->ctx_.device = Device::cuda(cudaDevice_);   // -1: every visible GPU
        } else {
            job->ctx_.device = Device::cpu();
        }
        job->ctx_.hpcDevice = hpcDevice_;
        job->ctx_.scratchDir = executor_.scratchDir();
        job->ctx_.hubToken = hubToken_ ? hubToken_() : std::string();
        // The worker itself is obtained by execute(), on the run's thread.
        job->backend_ = backend_;
        job->needsWorker_ = needsWorker;
        job->launcher_ = launcher_;
        job->remoteConfig_ = remote_;
        job->workerHint_ = workerHint_;
        job->nodeDatasets_ = engine_->datasets;
        job->uploads_ = std::move(uploads);
        job->nodePaths_ = nodePaths_;
        activeRun_ = job;
        if (backend_ == Backend::Cuda && cudaDevice_ == kAllCudaDevices && cudaAvailable())
            logLine("Run to step " + Step::number(target) + " on CUDA · all " +
                    std::to_string(cudaDeviceCount()) + " GPUs");
        else if (backend_ == Backend::Hpc && remote_.hasEngine())
            logLine("Run to step " + Step::number(target) + " on HPC · " + (remote_.where.empty() ? std::string("the cluster node") : remote_.where) +
                    " · " + toString(hpcDevice_));
        else if (backend_ == Backend::Hpc)
            logLine("Run to step " + Step::number(target) + " on HPC" + (remote_.known ? " (Python steps on the worker job)" : std::string()) + " · " +
                    toString(hpcDevice_));
        else
            logLine("Run to step " + Step::number(target) + " on " + toString(backend_));
        notify(&Observer::runStateChanged);
        return job;
    }

    void Workbench::finishRun(const std::shared_ptr<RunJob>& job) {
        if (!job) return;
        if (activeRun_ == job) activeRun_.reset();
        if (!job->finished()) {
            // never executed (or still executing, which the caller must not do)
            job->cancel();
            logLine("Run abandoned before it finished");
            notify(&Observer::runStateChanged);
            return;
        }
        if (!job->workerNote_.empty()) logLine(job->workerNote_);
        // Reports name steps by id: the pipeline is frozen during a run, but
        // an index would still be the wrong thing to trust here.
        for (const StepReport& r : job->reports()) {
            const int index = pipeline_.indexOf(r.id);
            const std::string name = index >= 0 ? Step::number(index) + " " + pipeline_.at(index).name
                                                : "(removed step " + Step::number(r.index) + ")";
            switch (r.state) {
                case StepReport::State::Failed: logLine("Step " + name + " failed: " + r.error); break;
                case StepReport::State::Ran: {
                    char buf[32];
                    std::snprintf(buf, sizeof buf, "%.1f s", r.seconds);
                    // where it ran, as the output says: the node's engine, the Python worker, this computer
                    std::string where;
                    if (r.index == 0 && job->onEngine_) where = " · on " + job->where_;   // the node opened it for the run
                    else if (index >= 0)
                        if (const std::optional<Executor::Held> h = executor_.held(r.id); h && h->output) where = " · " + placementText(*h->output);
                    logLine("Step " + name + " · " + buf + (r.note.empty() ? "" : " · " + r.note) + where);
                    if (session_.recording()) {
                        nlohmann::json entry{{"index", index}, {"seconds", r.seconds}, {"note", r.note}};
                        if (index >= 0) {
                            entry["kind"] = pipeline_.at(index).kind;
                            entry["params"] = pipeline_.at(index).params.toJson();
                            if (auto out = output(index)) {
                                entry["dims"] = out->meta.dims.toString();
                                if (out->labels && !out->labels->empty()) entry["labels"] = out->labels->stats().size();
                            }
                        }
                        session_.record("step_ran", entry);
                    }
                    break;
                }
                case StepReport::State::Skipped: logLine("Step " + name + " skipped"); break;
                case StepReport::State::Cached:
                case StepReport::State::Running: break;
            }
        }
        if (job->succeeded()) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.1f s", job->seconds());
            logLine(std::string("Run finished in ") + buf);
        } else {
            logLine("Run " + (job->wasCancelled() ? std::string("cancelled") : "failed: " + job->error()));
        }
        job->ownedRemote_.reset();
        // What the node holds now belongs to the session that answered; the
        // uploads are on the node for the next run.
        for (const auto& [key, node] : job->nodePaths_) nodePaths_[key] = node;
        if (!job->engineSession_.empty() && job->engineSession_ != engineSession_) {
            if (!engineSession_.empty())
                nodeOutputsGone(engineSession_, "held by an earlier SIRIUS engine (another cluster job), which has ended");
            engineSession_ = job->engineSession_;
        }
        // A run that re-ran the Load step (its tile, page order or voxel size
        // was edited) opened the data anew: that is the dataset from now on,
        // the output step 01 shows and the one the caches are seeded with
        // when they are cleared.
        bool reopened = false;
        if (source_) {
            std::shared_ptr<const StepOutput> load = executor_.cached(pipeline_, 0);
            if (load && load->source && load->source != source_) {
                loadOutput_ = load;
                loadOutputParams_ = pipeline_.at(0).params;
                source_ = load->source;
                datasetMeta_ = load->meta;
                reopened = true;
            }
        }
        if (reopened) notify(&Observer::datasetChanged);
        const DatasetMeta meta = outputMetaOf(viewed_);
        view_.channelVisible.resize(static_cast<std::size_t>(std::max<Index>(meta.dims.c, 1)), true);
        // z and t stay within the data on screen (displayedMeta). The output
        // is held across the statistics below, so a spilled array comes off
        // the disk once for both.
        const std::shared_ptr<const StepOutput> shown = displayOutput();
        const Dims5 dims = shown ? shown->meta.dims : meta.dims;
        view_.z = std::clamp<Index>(view_.z, 0, std::max<Index>(dims.z - 1, 0));
        view_.t = std::clamp<Index>(view_.t, 0, std::max<Index>(dims.t - 1, 0));
        syncLabelStats();   // a tracking step leaves the last frame's table
        notify(&Observer::outputsChanged);
        notify(&Observer::viewStateChanged);
        notify(&Observer::runStateChanged);
    }

    void Workbench::cancelRun() {
        if (activeRun_) {
            activeRun_->cancel();
            logLine("Cancelling…");
        }
    }

    // --- labels ------------------------------------------------------------------

    std::shared_ptr<LabelVolume> Workbench::viewedLabels() const {
        auto out = displayOutput();
        return out ? out->labels : nullptr;
    }

    std::shared_ptr<LabelVolume> Workbench::labelsOf(StepId id) const {
        if (pipeline_.indexOf(id) <= 0) return nullptr;   // Load has none; a removed step neither
        return executor_.lastLabels(id);
    }

    std::shared_ptr<LabelVolume> Workbench::editableLabels(StepId* id) {
        int actual = -1;
        auto out = displayOutput(&actual);
        if (id) *id = (out && actual >= 0) ? pipeline_.at(actual).id : 0;
        return out ? out->labels : nullptr;
    }

    void Workbench::notifyLabels(StepId id) {
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) o->labelsChanged(id);
    }

    void Workbench::applyLabelDiff(StepId id, const std::weak_ptr<LabelVolume>& target, const LabelDiff& diff, bool forward) {
        // The edit belongs to one label volume; when the step was re-run
        // (new labels) or removed since, there is nothing to undo into.
        std::shared_ptr<LabelVolume> current = labelsOf(id);
        std::shared_ptr<LabelVolume> expected = target.lock();
        if (!current || !expected || current != expected) {
            const int index = pipeline_.indexOf(id);
            logLine(std::string(forward ? "Redo" : "Undo") + " of a label edit skipped: the labels of " +
                    (index >= 0 ? "step " + Step::number(index) + " " + pipeline_.at(index).name : "a removed step") +
                    " have been recomputed since.");
            return;
        }
        current->apply(diff, forward);
        // The statistics describe one frame. A diff of another one -- an
        // undo after the view moved on from the edit's frame -- leaves them
        // where they are (that frame is measured again when it is shown);
        // updating from it used to move the table to the edit's frame.
        if (current->statsT() == diff.t || current->statsT() < 0) current->updateStats(diff);
        staleBelow(id);
        syncLabelStats();   // the table on the viewed frame, whatever it was on
        notifyLabels(id);
        notify(&Observer::outputsChanged);
    }

    void Workbench::staleBelow(StepId id) {
        const int index = pipeline_.indexOf(id);
        if (index < 0) return;
        for (int j = index + 1; j < pipeline_.size(); ++j) executor_.markStale(pipeline_.at(j).id);
    }

    void Workbench::pushLabelCommand(const std::string& label, const std::string& mergeKey, StepId id,
                                     const std::shared_ptr<LabelVolume>& labels, std::shared_ptr<LabelDiff> diff) {
        std::weak_ptr<LabelVolume> target = labels;
        Command c;
        c.label = label;
        c.mergeKey = mergeKey;
        c.undo = [this, id, target, diff] { applyLabelDiff(id, target, *diff, false); };
        c.redo = [this, id, target, diff] { applyLabelDiff(id, target, *diff, true); };
        pushCommand(std::move(c));
    }

    void Workbench::recordLabelDiffs(const std::string& label, StepId id, const std::shared_ptr<LabelVolume>& labels,
                                     std::vector<LabelDiff> diffs) {
        diffs.erase(std::remove_if(diffs.begin(), diffs.end(), [](const LabelDiff& d) { return d.empty(); }), diffs.end());
        if (diffs.empty() || !labels) return;
        std::size_t voxels = 0;
        for (const LabelDiff& d : diffs) voxels += d.indices.size();
        session_.record("label_edit", {{"what", label}, {"voxels", voxels}, {"frames", diffs.size()}});
        // only the frame the table is on needs its statistics brought up to
        // date; measuring every other frame on the way costs a full scan each
        for (const LabelDiff& d : diffs)
            if (labels->statsT() == d.t || labels->statsT() < 0) labels->updateStats(d);
        auto shared = std::make_shared<std::vector<LabelDiff>>(std::move(diffs));
        std::weak_ptr<LabelVolume> target = labels;
        Command c;
        c.label = label;
        c.undo = [this, id, target, shared] {
            for (auto it = shared->rbegin(); it != shared->rend(); ++it) applyLabelDiff(id, target, *it, false);
        };
        c.redo = [this, id, target, shared] {
            for (const LabelDiff& d : *shared) applyLabelDiff(id, target, d, true);
        };
        pushCommand(std::move(c));
        staleBelow(id);
        syncLabelStats();   // the table on the viewed frame
        notifyLabels(id);
        notify(&Observer::outputsChanged);
    }

    void Workbench::recordLabelDiff(const std::string& label, StepId id, const std::shared_ptr<LabelVolume>& labels,
                                    LabelDiff diff) {
        if (diff.empty() || !labels) return;
        session_.record("label_edit", {{"what", label}, {"voxels", diff.indices.size()}, {"t", diff.t}});
        labels->updateStats(diff);
        pushLabelCommand(label, {}, id, labels, std::make_shared<LabelDiff>(std::move(diff)));
        staleBelow(id);
        notifyLabels(id);
        notify(&Observer::outputsChanged);
    }

    void Workbench::beginPaintStroke() {
        endPaintStroke();
        if (refuseIfRunning("paint labels")) return;
        ++strokeCounter_;
        strokeDiff_ = LabelDiff{};
        strokeLabels_ = editableLabels(&strokeStep_);
        strokeOpen_ = static_cast<bool>(strokeLabels_);
        // The lasso paints with the brush for now, and so paints a new label
        // too: label 0 would erase whatever the stroke crosses.
        if (view_.selectedLabel == 0 && strokeLabels_ &&
            (view_.paintTool == PaintTool::Brush || view_.paintTool == PaintTool::Lasso))
            view_.selectedLabel = strokeLabels_->maxLabel() + 1;
        // the stroke's bounds: what a replay needs to group the paint events
        // between them (each move records one), and what undo works on
        if (strokeOpen_ && session_.recording())
            session_.record("stroke_begin", {{"stroke", strokeCounter_}, {"step", pipeline_.indexOf(strokeStep_)}, {"t", view_.t}, {"label", view_.selectedLabel}, {"tool", toString(view_.paintTool)}, {"brush_px", view_.brushPx}, {"paint_3d", view_.paint3d}});
    }

    void Workbench::paintLabels(Index z, Index y, Index x, bool erase) {
        static const bool trace = std::getenv("SIRIUS_TRACE_VIEW") != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        if (!strokeOpen_) beginPaintStroke();
        if (!strokeOpen_) return;   // nothing to paint on, or a run is active
        // The stroke edits the volume it started on: a display change in the
        // middle of a drag must not spill the stroke into another output.
        const std::shared_ptr<LabelVolume>& labels = strokeLabels_;
        const auto t1 = std::chrono::steady_clock::now();
        struct Report {
            bool on;
            std::chrono::steady_clock::time_point t0, t1;
            ~Report() {
                if (!on) return;
                const auto t2 = std::chrono::steady_clock::now();
                std::fprintf(stderr, "paintLabels: lookup %lld us · edit+notify %lld us\n",
                             static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()),
                             static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()));
            }
        } report{trace, t0, t1};
        const double radius = std::max(1.0, view_.brushPx / 2.0);
        const Index zRadius = view_.paint3d ? paintZRadius(view_.brushPx) : 0;
        const std::uint32_t label = erase ? 0u : view_.selectedLabel;
        LabelDiff diff = labels->paint(view_.t, z, y, x, radius, zRadius, label, erase ? view_.selectedLabel : 0u);
        if (diff.empty()) return;
        session_.record("paint", {{"z", z}, {"y", y}, {"x", x}, {"t", view_.t}, {"erase", erase}, {"label", label}, {"brush_px", view_.brushPx}, {"paint_3d", view_.paint3d}, {"voxels", diff.indices.size()}});
        // One stroke = one undo entry: the accumulated diff replaces the entry
        // pushed by the previous mouse move (History merges by key). The
        // statistics wait for endPaintStroke() so every move stays cheap.
        strokeDiff_.t = diff.t;
        strokeDiff_.indices.insert(strokeDiff_.indices.end(), diff.indices.begin(), diff.indices.end());
        strokeDiff_.before.insert(strokeDiff_.before.end(), diff.before.begin(), diff.before.end());
        strokeDiff_.after.insert(strokeDiff_.after.end(), diff.after.begin(), diff.after.end());
        pushLabelCommand(erase ? "Erase labels" : "Paint label " + std::to_string(label),
                         "stroke#" + std::to_string(strokeCounter_), strokeStep_, labels,
                         std::make_shared<LabelDiff>(strokeDiff_));
        staleBelow(strokeStep_);   // cheap; the outputsChanged notification waits for the stroke's end
        notifyLabels(strokeStep_);
    }

    int Workbench::paintZRadius(int brushPx) noexcept { return std::max(1, brushPx / 6); }

    void Workbench::endPaintStroke() {
        if (!strokeOpen_) return;
        strokeOpen_ = false;
        std::shared_ptr<LabelVolume> labels = std::move(strokeLabels_);
        strokeLabels_.reset();
        if (session_.recording()) {
            nlohmann::json end{{"stroke", strokeCounter_}, {"t", strokeDiff_.t}, {"voxels", strokeDiff_.indices.size()}};
            if (labels && !strokeDiff_.empty()) {
                // the box the stroke touched (half open), so a replay or a
                // training crop knows where to look without the moves
                const Index ly = labels->y(), lx = labels->x();
                Index z0 = labels->z(), z1 = 0, y0 = ly, y1 = 0, x0 = lx, x1 = 0;
                for (Index i : strokeDiff_.indices) {
                    const Index z = i / (ly * lx), y = (i / lx) % ly, x = i % lx;
                    z0 = std::min(z0, z), z1 = std::max(z1, z + 1);
                    y0 = std::min(y0, y), y1 = std::max(y1, y + 1);
                    x0 = std::min(x0, x), x1 = std::max(x1, x + 1);
                }
                end["bbox"] = {z0, z1, y0, y1, x0, x1};
            }
            session_.record("stroke_end", end);
        }
        if (!labels || strokeDiff_.empty()) return;
        labels->updateStats(strokeDiff_);
        strokeDiff_ = LabelDiff{};
        notifyLabels(strokeStep_);
        notify(&Observer::outputsChanged);
    }

    void Workbench::fillLabel(Index z, Index y, Index x) {
        endPaintStroke();
        if (refuseIfRunning("fill a label")) return;
        StepId id = 0;
        auto labels = editableLabels(&id);
        if (!labels) return;
        const std::uint32_t label = view_.selectedLabel ? view_.selectedLabel : labels->maxLabel() + 1;
        recordLabelDiff("Fill label " + std::to_string(label), id, labels, labels->fill(view_.t, z, y, x, label));
    }

    void Workbench::mergeLabels(const std::vector<std::uint32_t>& ids) {
        endPaintStroke();
        if (ids.size() < 2 || refuseIfRunning("merge labels")) return;
        StepId id = 0;
        auto labels = editableLabels(&id);
        if (!labels) return;
        if (labels->tracked() && labels->t() > 1) {
            std::vector<LabelDiff> diffs;
            for (Index t = 0; t < labels->t(); ++t) diffs.push_back(labels->merge(t, ids));
            recordLabelDiffs("Merge tracks", id, labels, std::move(diffs));
            return;
        }
        recordLabelDiff("Merge labels", id, labels, labels->merge(view_.t, ids));
    }

    namespace {
        // The rounded centroid of `id` in frame t, or {-1, -1, -1} when absent.
        std::array<Index, 3> centroidOf(const LabelVolume& labels, Index t, std::uint32_t id) {
            const std::uint32_t* v = labels.volume(t);
            double sz = 0, sy = 0, sx = 0;
            Index n = 0;
            for (Index z = 0; z < labels.z(); ++z)
                for (Index y = 0; y < labels.y(); ++y) {
                    const std::uint32_t* row = v + (z * labels.y() + y) * labels.x();
                    for (Index x = 0; x < labels.x(); ++x)
                        if (row[x] == id) {
                            sz += static_cast<double>(z);
                            sy += static_cast<double>(y);
                            sx += static_cast<double>(x);
                            ++n;
                        }
                }
            if (!n) return {-1, -1, -1};
            return {static_cast<Index>(std::lround(sz / n)), static_cast<Index>(std::lround(sy / n)), static_cast<Index>(std::lround(sx / n))};
        }

        // Moves `seed` onto the voxel of `id` nearest to it in frame t (the
        // centroid of a bent object lies outside it); false when `id` is absent,
        // or the labels have no frame t (a view on a time point past them).
        bool snapToLabel(const LabelVolume& labels, Index t, std::uint32_t id, std::array<Index, 3>& seed) {
            if (t < 0 || t >= labels.t()) return false;
            const std::uint32_t* v = labels.volume(t);
            // A seed on the label stays, without a pass over the volume: the
            // viewer's Split tool only takes clicks on the label, and a
            // centroid usually lies inside its object.
            if (seed[0] >= 0 && seed[0] < labels.z() && seed[1] >= 0 && seed[1] < labels.y() && seed[2] >= 0 &&
                seed[2] < labels.x() && v[(seed[0] * labels.y() + seed[1]) * labels.x() + seed[2]] == id)
                return true;
            double best = std::numeric_limits<double>::infinity();
            std::array<Index, 3> nearest{-1, -1, -1};
            for (Index z = 0; z < labels.z(); ++z)
                for (Index y = 0; y < labels.y(); ++y) {
                    const std::uint32_t* row = v + (z * labels.y() + y) * labels.x();
                    for (Index x = 0; x < labels.x(); ++x) {
                        if (row[x] != id) continue;
                        const double dz = static_cast<double>(z - seed[0]), dy = static_cast<double>(y - seed[1]),
                                     dx = static_cast<double>(x - seed[2]);
                        const double d = dz * dz + dy * dy + dx * dx;
                        if (d < best) {
                            best = d;
                            nearest = {z, y, x};
                        }
                    }
                }
            if (nearest[0] < 0) return false;
            seed = nearest;
            return true;
        }
    } // namespace

    void Workbench::splitLabel(std::uint32_t label, std::array<Index, 3> a, std::array<Index, 3> b) {
        endPaintStroke();
        if (label == 0 || refuseIfRunning("split a label")) return;
        StepId id = 0;
        auto labels = editableLabels(&id);
        if (!labels) return;
        // The seeds go onto the label's nearest voxels: points picked from its
        // bounding box, or clicked just beside it, fall outside a bent, ring
        // or diagonal object, and the split would refuse them.
        if (!snapToLabel(*labels, view_.t, label, a) || !snapToLabel(*labels, view_.t, label, b)) {
            logLine("Split: label " + std::to_string(label) + " is not in this frame.");
            return;
        }
        if (a == b) {
            logLine("Split: pick two different points inside label " + std::to_string(label) + ".");
            return;
        }
        if (labels->tracked() && labels->t() > 1) {
            // The split travels along the track: the two parts' centroids in
            // one frame seed the same watershed of the same id in the next,
            // forward and backward, so the new part keeps one id throughout.
            std::vector<LabelDiff> diffs;
            LabelDiff first = labels->split(view_.t, label, a, b);
            if (first.empty()) return;
            const std::uint32_t part = labels->maxLabel();
            diffs.push_back(std::move(first));
            for (const Index dir : {Index{1}, Index{-1}}) {
                std::array<Index, 3> seedA = centroidOf(*labels, view_.t, label), seedB = centroidOf(*labels, view_.t, part);
                for (Index f = view_.t + dir; f >= 0 && f < labels->t(); f += dir) {
                    if (!snapToLabel(*labels, f, label, seedA)) continue;   // the object is absent here
                    if (!snapToLabel(*labels, f, label, seedB) || seedA == seedB) continue;
                    LabelDiff d = labels->split(f, label, seedA, seedB, part);
                    if (d.empty()) continue;
                    diffs.push_back(std::move(d));
                    seedA = centroidOf(*labels, f, label);
                    seedB = centroidOf(*labels, f, part);
                }
            }
            recordLabelDiffs("Split track " + std::to_string(label), id, labels, std::move(diffs));
            return;
        }
        recordLabelDiff("Split label " + std::to_string(label), id, labels, labels->split(view_.t, label, a, b));
    }

    void Workbench::deleteLabel(std::uint32_t label) {
        endPaintStroke();
        if (label == 0 || refuseIfRunning("delete a label")) return;
        StepId id = 0;
        auto labels = editableLabels(&id);
        if (!labels) return;
        if (view_.selectedLabel == label) view_.selectedLabel = 0;
        if (labels->tracked() && labels->t() > 1) {
            std::vector<LabelDiff> diffs;
            for (Index t = 0; t < labels->t(); ++t) diffs.push_back(labels->remove(t, label));
            recordLabelDiffs("Delete track " + std::to_string(label), id, labels, std::move(diffs));
            return;
        }
        LabelDiff diff = labels->remove(view_.t, label);
        recordLabelDiff("Delete label " + std::to_string(label), id, labels, std::move(diff));
    }

    void Workbench::setLabelReviewed(std::uint32_t label, bool reviewed) {
        endPaintStroke();
        if (refuseIfRunning("mark a label reviewed")) return;
        StepId id = 0;
        auto labels = editableLabels(&id);
        if (!labels) return;
        for (LabelStats& s : labels->stats())
            if (s.id == label) s.reviewed = reviewed;
        session_.record("review", {{"step", pipeline_.indexOf(id)}, {"label", label}, {"reviewed", reviewed}});
        notifyLabels(id);
    }

    void Workbench::acceptAllReviewed() {
        endPaintStroke();
        if (refuseIfRunning("accept the labels")) return;
        StepId id = 0;
        auto labels = editableLabels(&id);
        if (!labels) return;
        for (LabelStats& s : labels->stats()) s.reviewed = true;
        session_.record("review_all", {{"step", pipeline_.indexOf(id)}, {"labels", labels->stats().size()}});
        logLine("Accepted all reviewed labels");
        notifyLabels(id);
    }

    std::uint32_t Workbench::nextFlaggedLabel(bool forward) {
        endPaintStroke();
        syncLabelStats();
        auto labels = viewedLabels();
        if (!labels) return 0;
        const auto& stats = labels->stats();
        if (stats.empty()) return 0;
        const int n = static_cast<int>(stats.size());
        int start = 0;
        for (int i = 0; i < n; ++i)
            if (stats[static_cast<std::size_t>(i)].id == view_.selectedLabel) start = i;
        for (int k = 1; k <= n; ++k) {
            const int i = ((start + (forward ? k : -k)) % n + n) % n;
            const LabelStats& s = stats[static_cast<std::size_t>(i)];
            if (s.flags.empty() || s.reviewed) continue;
            view_.selectedLabel = s.id;
            view_.cx = (s.bbox[4] + s.bbox[5]) / 2;
            view_.cy = (s.bbox[2] + s.bbox[3]) / 2;
            view_.z = (s.bbox[0] + s.bbox[1]) / 2;
            view_.labels = true;
            notify(&Observer::viewStateChanged);
            return s.id;
        }
        return 0;
    }

} // namespace sirius::app
