#include "core/engine_node.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <sirius/device.hpp>

#include "core/array_source.hpp"
#include "core/cancel.hpp"
#include "core/executor.hpp"
#include "core/host.hpp"
#include "core/operation.hpp"
#include "core/ops/builtin.hpp"
#include "core/pipeline.hpp"
#include "core/remote_source.hpp"
#include "core/serialize.hpp"
#include "core/sha256.hpp"
#include "core/statistics.hpp"

namespace sirius::app {

    namespace fs = std::filesystem;
    using json = nlohmann::json;

    namespace {

        // Most tensors one reply frame may carry (rpc::kMaxTensors), less a margin.
        constexpr std::size_t kMaxReplyTensors = rpc::kMaxTensors - 4;
        // A chunk of an upload: the application sends 8 MiB.
        constexpr std::size_t kMaxChunk = std::size_t{64} << 20;

        json localized(const json& v) {
            if (v.is_string()) {
                std::string host, path;
                if (splitClusterPath(v.get<std::string>(), host, path)) return path;
                return v;
            }
            if (v.is_array()) {
                json a = json::array();
                for (const json& e : v) a.push_back(localized(e));
                return a;
            }
            return v;
        }

        std::string intText(std::int64_t v) { return std::to_string(v); }

        std::string stringParam(const json& p, const char* key) {
            const auto it = p.find(key);
            return it != p.end() && it->is_string() ? it->get<std::string>() : std::string();
        }

        // The node's meta of every step up to `index` (what Workbench::inputMetaOf does on the application's side).
        DatasetMeta inputMetaOf(const Pipeline& p, int index) {
            DatasetMeta meta;
            for (int i = 0; i < std::min(index, p.size()); ++i) {
                const Step& s = p.at(i);
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

        Validation validateStep(const Pipeline& p, int index) {
            Validation v;
            const Step& s = p.at(index);
            const Operation* op = findOperation(s.kind);
            if (!op) {
                v.errors.push_back("the engine has no operation '" + s.kind + "'");
                return v;
            }
            try {
                return op->validate(s.params, inputMetaOf(p, index));
            } catch (const std::exception& e) {
                v.errors.push_back(e.what());
            }
            return v;
        }

        // A file name as the upload keeps it: the last component, nothing that walks.
        std::string safeName(const std::string& name) {
            std::string base = fs::u8path(name).filename().u8string();
            if (base.empty() || base == "." || base == ".." || base.find_first_of("/\\:") != std::string::npos) return "upload";
            return base;
        }

        bool safeKey(const std::string& key) {
            return !key.empty() && key.size() <= 64 && key.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyz") == std::string::npos;
        }

        std::string fileSha256(const fs::path& path) {
            std::ifstream in(path, std::ios::binary);
            if (!in) throw std::runtime_error("cannot read " + path.u8string());
            crypto::Sha256 h;
            std::vector<char> buf(std::size_t{4} << 20);
            while (in) {
                in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
                const std::streamsize n = in.gcount();
                if (n > 0) h.update(buf.data(), static_cast<std::size_t>(n));
            }
            return crypto::toHex(h.finish());
        }

        // What the context runs on for a request's "device": the engine's GPU
        // ("cuda", "cuda:N"; "cuda:all" = every GPU of the job, volumes
        // round-robined) or its CPU. "cuda" on a node without one is the CPU.
        void chooseDevice(const std::string& requested, const std::string& fallback, StepContext& ctx, std::string& said) {
            std::string d = requested.empty() ? fallback : requested;
            for (char& c : d) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (d == "auto") d = "cuda";
            if (d.rfind("cuda", 0) == 0 && cudaAvailable() && cudaDeviceCount() > 0) {
                ctx.backend = Backend::Cuda;
                int index = 0;
                if (d == "cuda:all") index = -1;
                else if (d.rfind("cuda:", 0) == 0) index = std::clamp(std::atoi(d.c_str() + 5), 0, cudaDeviceCount() - 1);
                ctx.device = Device::cuda(index);
                said = index < 0 ? "cuda:all" : "cuda:" + std::to_string(index);
                return;
            }
            ctx.backend = Backend::Cpu;
            ctx.device = Device::cpu();
            said = d.rfind("cuda", 0) == 0 ? "cpu (the node has no GPU)" : "cpu";
        }

    } // namespace

    json nodePipelineJson(const json& pipeline) {
        json out = pipeline;
        if (!out.contains("steps") || !out["steps"].is_array()) return out;
        for (json& s : out["steps"])
            if (s.is_object() && s.contains("params") && s["params"].is_object())
                for (auto& [key, value] : s["params"].items()) value = localized(value);
        return out;
    }

    struct EngineNode::Impl {
        Options options;
        std::string session;
        fs::path scratch;
        bool ownScratch = false;
        std::unique_ptr<Executor> executor;
        // uploads in progress: key -> bytes written so far
        std::mutex uploadMutex;
        std::map<std::string, std::uint64_t> uploads;

        explicit Impl(Options o) : options(std::move(o)) {
            registerBuiltinOperations();
            session = rpc::randomNonce().substr(0, 16);
            std::error_code ec;
            if (options.scratch.empty()) {
                scratch = fs::temp_directory_path(ec) / ("sirius-engine-" + session);
                ownScratch = true;
            } else {
                scratch = options.scratch / ("sirius-engine-" + session);
                ownScratch = true;
            }
            fs::create_directories(scratch / "uploads", ec);
            executor = std::make_unique<Executor>(scratch / "cache");
        }

        ~Impl() {
            executor.reset();
            if (ownScratch) {
                std::error_code ec;
                fs::remove_all(scratch, ec);
            }
        }

        void log(const std::string& line) const {
            if (options.log) options.log(line);
        }

        Pipeline pipelineOf(const json& params) const {
            if (!params.contains("pipeline") || !params["pipeline"].is_object()) throw std::runtime_error("no pipeline");
            return Pipeline::fromJson(nodePipelineJson(params["pipeline"]));
        }

        StepContext contextFor(const json& params, std::string& device) const {
            StepContext ctx;
            chooseDevice(stringParam(params, "device"), options.defaultDevice, ctx, device);
            ctx.hubToken = stringParam(params, "hub_token");
            ctx.scratchDir = scratch / "run";
            std::error_code ec;
            fs::create_directories(ctx.scratchDir, ec);
            return ctx;
        }

        std::shared_ptr<const StepOutput> outputOf(const std::string& handle) const {
            std::string s, fp;
            std::uint64_t step = 0;
            if (!parseOutputHandle(handle, s, step, fp)) throw DatasetError(handle + " is not a step output's handle");
            if (s != session)
                throw DatasetError("this result was held by an earlier SIRIUS engine (another cluster job, or one that was restarted), which has "
                                   "ended: run the step again");
            std::shared_ptr<const StepOutput> out = executor->outputWithFingerprint(step, fp);
            if (!out) {
                const std::optional<Executor::Held> h = executor->held(step);
                if (h && h->fingerprint == fp)
                    throw DatasetError("this result is no longer kept on the node (the step's cache is Recompute and a later step "
                                       "ran): run the step again, or set its cache to Memory or Disk");
                throw DatasetError("this result was replaced on the node (the step ran again with other parameters): run it again");
            }
            return out;
        }
    };

    EngineNode::EngineNode(Options options) : impl_(std::make_unique<Impl>(std::move(options))) {}
    EngineNode::~EngineNode() = default;

    const std::string& EngineNode::session() const noexcept { return impl_->session; }
    const fs::path& EngineNode::scratch() const noexcept { return impl_->scratch; }

    DatasetService::ResolvedOutput EngineNode::resolve(const std::string& handle) const {
        const std::shared_ptr<const StepOutput> out = impl_->outputOf(handle);
        DatasetService::ResolvedOutput r;
        r.meta = out->meta;
        if (out->array) r.source = std::make_shared<MemorySource>(out->array, out->meta);
        else r.source = out->source;
        if (!r.source) throw DatasetError("the step's output holds no data on the node");
        return r;
    }

    json EngineNode::cacheStatus() const {
        return {{"bytes", impl_->executor->cachedBytes()}, {"session", impl_->session}, {"scratch", impl_->scratch.u8string()}};
    }

    // --- pipeline_run ---------------------------------------------------------------------------------

    rpc::Reply EngineNode::pipelineRun(const rpc::Request& req, rpc::CallContext& call) {
        Impl& d = *impl_;
        const json& params = req.params;
        const Pipeline p = d.pipelineOf(params);
        int target = params.contains("target") && params["target"].is_number_integer() ? params["target"].get<int>() : p.size() - 1;
        if (target < 0 || target >= p.size()) target = p.size() - 1;
        std::set<std::string> have;
        if (params.contains("have") && params["have"].is_array())
            for (const json& h : params["have"])
                if (h.is_string()) have.insert(h.get<std::string>());

        // Validated here too, with the node's files: a flat-field image the
        // application only named (cluster://) is read here for the first time.
        for (int i = 1; i <= target; ++i) {
            const Step& s = p.at(i);
            if (!s.enabled) continue;
            const Validation v = validateStep(p, i);
            if (!v.ok())
                throw std::runtime_error("Step " + Step::number(i) + " " + s.name + " cannot run on " + host::hostName() + ": " + v.firstError());
        }

        std::string device;
        StepContext ctx = d.contextFor(params, device);
        // The Python worker only when a step that will run needs it.
        std::unique_ptr<RemoteWorker> python;
        int toRun = 0;
        for (int i = 0; i <= target; ++i) {
            const Step& s = p.at(i);
            if ((i == 0 || s.enabled) && !d.executor->isFresh(p, i)) {
                ++toRun;
                if (i > 0 && !python && s.op().needsWorker(s.params)) {
                    if (!d.options.takePython)
                        throw std::runtime_error("Step " + Step::number(i) + " " + s.name +
                                                 " needs the Python worker, and this engine runs without one (sirius-cli serve --no-python-worker)");
                    python = d.options.takePython(call.cancelFlag());
                }
            }
        }
        ctx.remote = python.get();
        int done = 0, current = -1;
        ctx.progress = [&](double f, const std::string& m) {
            const double overall = toRun > 0 ? (done + std::clamp(f, 0.0, 1.0)) / toRun : 1.0;
            call.progress(overall, m, {{"step", current}, {"state", "running"}});
        };
        ctx.cancelled = call.cancelFlag();
        std::vector<StepReport> reports;
        auto onStep = [&](const StepReport& r) {
            if (r.state == StepReport::State::Running) current = r.index;
            else if (r.state == StepReport::State::Ran) ++done;
            const double overall = toRun > 0 ? static_cast<double>(done) / toRun : 1.0;
            json extra = {{"step", r.index}, {"state", toString(r.state)}};
            if (r.state == StepReport::State::Ran) extra["seconds"] = r.seconds;
            call.progress(overall, "", extra);
        };
        std::string error;
        try {
            d.executor->run(p, target, ctx, &reports, onStep);
        } catch (const CancelledError&) {
            if (python && d.options.giveBackPython) d.options.giveBackPython(std::move(python));
            throw;
        } catch (const std::exception& e) {
            if (isCancellation(e) || call.cancelled()) {
                if (python && d.options.giveBackPython) d.options.giveBackPython(std::move(python));
                throw CancelledError();
            }
            error = e.what();
        }
        ctx.remote = nullptr;
        if (python && d.options.giveBackPython) d.options.giveBackPython(std::move(python));

        rpc::Reply reply;
        json outputs = json::array();
        for (int i = 0; i <= target; ++i) {
            const Step& s = p.at(i);
            if (i > 0 && !s.enabled) continue;
            const std::optional<Executor::Held> h = d.executor->held(s.id);
            const std::string fp = d.executor->fingerprint(p, i);
            if (!h || !h->output || h->fingerprint != fp) continue;
            const StepOutput& out = *h->output;
            const std::string handle = makeOutputHandle(d.session, s.id, fp);
            json o = {{"index", i},
                      {"step_id", s.id},
                      {"handle", handle},
                      {"fingerprint", fp},
                      {"held", h->data},
                      {"meta", toJson(out.meta)},
                      {"note", out.note},
                      {"seconds", out.seconds},
                      {"cache", toString(s.cache)},
                      {"bytes", h->bytes},
                      {"ran_on", {{"backend", toString(out.ranOn)}, {"device", out.ranOnDevice}}},
                      {"labels", {{"present", out.labels && !out.labels->empty()}, {"count", out.labels ? out.labels->stats().size() : std::size_t{0}}}}};
            if (!have.count(handle) && !out.diagnostics.empty()) {
                EncodedDiagnostics e = encodeDiagnostics(out.diagnostics, "s" + intText(i) + "_");
                if (reply.tensors.size() + e.tensors.size() > kMaxReplyTensors) {
                    // more images than one frame carries: the rest stay on the node
                    Diagnostics lean = out.diagnostics;
                    lean.images.clear();
                    lean.warnings.push_back("The diagnostic images stayed on the node: there were too many to send at once.");
                    e = encodeDiagnostics(lean, "s" + intText(i) + "_");
                }
                o["diagnostics"] = std::move(e.json);
                for (rpc::Tensor& t : e.tensors) reply.tensors.push_back(std::move(t));
            }
            outputs.push_back(std::move(o));
        }
        json reportList = json::array();
        for (const StepReport& r : reports) reportList.push_back(toJson(r));
        reply.result = {{"session", d.session}, {"device", device}, {"reports", reportList}, {"outputs", outputs}};
        if (!error.empty()) reply.result["error"] = error;
        return reply;
    }

    // --- step_preview / step_validate -------------------------------------------------------------------

    rpc::Reply EngineNode::stepPreview(const rpc::Request& req, rpc::CallContext& call) {
        Impl& d = *impl_;
        const Pipeline p = d.pipelineOf(req.params);
        const int index = req.params.value("index", -1);
        if (index < 1 || index >= p.size()) throw std::runtime_error("step_preview: no step " + std::to_string(index));
        const Step& s = p.at(index);
        ParamSet params = s.params;
        if (req.params.contains("params") && req.params["params"].is_object()) {
            json given = req.params["params"];
            for (auto& [key, value] : given.items()) value = localized(value);
            params = ParamSet::fromJson(given);
            params.applyDefaults(s.op().info().params);
            params.coerce(s.op().info().params);
        }
        // The input as the node holds it: the nearest enabled step above with
        // a fresh output, or the Load step opened for the purpose (lazily).
        int up = index - 1;
        while (up > 0 && !p.at(up).enabled) --up;
        std::shared_ptr<const StepOutput> input = d.executor->cached(p, up);
        if (!input) {
            if (up != 0)
                throw std::runtime_error("the input of step " + Step::number(index) + " (step " + Step::number(up) +
                                         ") is not computed on the node yet: run up to it");
            std::string device;
            StepContext ctx = d.contextFor(req.params, device);
            ctx.cancelled = call.cancelFlag();
            input = std::make_shared<StepOutput>(p.at(0).op().run(StepInput{}, p.at(0).params, ctx));
        }
        rpc::Reply reply;
        reply.result = {{"diagnostics", nullptr}};
        if (std::optional<Diagnostics> diag = s.op().preview(input->asInput(), params)) {
            EncodedDiagnostics e = encodeDiagnostics(*diag);
            reply.result["diagnostics"] = std::move(e.json);
            reply.tensors = std::move(e.tensors);
        }
        if (req.params.value("initial", false)) reply.result["initial_params"] = s.op().initialParams(params, input->asInput()).toJson();
        return reply;
    }

    rpc::Reply EngineNode::stepValidate(const rpc::Request& req) {
        const Pipeline p = impl_->pipelineOf(req.params);
        const int index = req.params.value("index", -1);
        if (index < 0 || index >= p.size()) throw std::runtime_error("step_validate: no step " + std::to_string(index));
        const Validation v = validateStep(p, index);
        rpc::Reply reply;
        reply.result = {{"errors", v.errors}, {"warnings", v.warnings}};
        return reply;
    }

    // --- output_stats ---------------------------------------------------------------------------------------

    rpc::Reply EngineNode::outputStats(const rpc::Request& req, rpc::CallContext& call) {
        const std::string path = stringParam(req.params, "path");
        if (path.empty()) throw std::runtime_error("output_stats: no path");
        StepOutput out;
        if (isOutputHandle(path)) {
            try {
                out = *impl_->outputOf(path);
            } catch (const DatasetError& e) {
                throw std::runtime_error(std::string("DatasetError: ") + e.what());
            }
        } else {
            // a dataset on the node, as the Load step's options shape it
            OpenOptions o;
            const json opts = req.params.contains("options") && req.params["options"].is_object() ? req.params["options"] : json::object();
            if (opts.contains("page_order") && opts["page_order"].is_string()) {
                PageOrder po;
                po.order = opts["page_order"].get<std::string>();
                po.c = opts.value("c", Index{0});
                po.t = opts.value("t", Index{0});
                po.z = opts.value("z", Index{0});
                o.pageOrder = po;
            }
            OpenResult opened = openDataset(path, o);
            out.meta = opened.meta;
            out.source = opened.source;
        }
        const StatisticsOptions o = statisticsOptionsFromJson(req.params.value("statistics", json::object()));
        const std::vector<ChannelStatistics> stats =
            channelStatistics(out, o, [&call](double f) { call.progress(f, "measuring"); }, call.cancelFlag());
        rpc::Reply reply;
        reply.result = {{"channels", channelStatisticsToJson(stats)}};
        return reply;
    }

    // --- uploads -----------------------------------------------------------------------------------------------

    rpc::Reply EngineNode::statFile(const rpc::Request& req) {
        const std::string key = stringParam(req.params, "key");
        if (!safeKey(key)) throw std::runtime_error("stat_file: a malformed key");
        const fs::path file = impl_->scratch / "uploads" / key / safeName(stringParam(req.params, "name"));
        std::error_code ec;
        const bool exists = fs::is_regular_file(file, ec);
        const std::uint64_t size = req.params.value("size", std::uint64_t{0});
        rpc::Reply reply;
        reply.result = {{"exists", exists && fs::file_size(file, ec) == size}};
        if (reply.result["exists"].get<bool>()) reply.result["path"] = file.generic_u8string();
        return reply;
    }

    rpc::Reply EngineNode::putFile(const rpc::Request& req) {
        Impl& d = *impl_;
        const std::string key = stringParam(req.params, "key");
        if (!safeKey(key)) throw std::runtime_error("put_file: a malformed key");
        const std::uint64_t size = req.params.value("size", std::uint64_t{0});
        const std::uint64_t offset = req.params.value("offset", std::uint64_t{0});
        if (size > d.options.maxUploadBytes) throw std::runtime_error("put_file: " + std::to_string(size) + " bytes is more than this engine accepts");
        const rpc::Tensor* data = nullptr;
        for (const rpc::Tensor& t : req.tensors)
            if (t.name == "data") data = &t;
        if (!data || data->bytes.size() > kMaxChunk) throw std::runtime_error("put_file: a chunk of 1 to 64 MiB is expected as the tensor \"data\"");
        const fs::path dir = d.scratch / "uploads" / key;
        const fs::path file = dir / safeName(stringParam(req.params, "name"));
        const fs::path part = fs::path(file.u8string() + ".part");
        const std::lock_guard<std::mutex> g(d.uploadMutex);
        std::error_code ec;
        fs::create_directories(dir, ec);
        std::uint64_t& written = d.uploads[key];
        if (offset == 0) {
            written = 0;
            fs::remove(part, ec);
        }
        if (offset != written) throw std::runtime_error("put_file: a chunk at " + std::to_string(offset) + " where " + std::to_string(written) + " was expected");
        if (offset + data->bytes.size() > size) throw std::runtime_error("put_file: more bytes than the file's size");
        {
            std::ofstream out(part, std::ios::binary | std::ios::app);
            out.write(reinterpret_cast<const char*>(data->bytes.data()), static_cast<std::streamsize>(data->bytes.size()));
            if (!out) throw std::runtime_error("put_file: cannot write " + part.u8string() + " (is the node's scratch full?)");
        }
        written += data->bytes.size();
        rpc::Reply reply;
        reply.result = {{"received", written}};
        if (written == size) {
            const std::string expect = stringParam(req.params, "sha256");
            if (!expect.empty() && fileSha256(part) != expect) {
                fs::remove(part, ec);
                d.uploads.erase(key);
                throw std::runtime_error("put_file: the upload arrived damaged (its SHA-256 differs): try again");
            }
            fs::rename(part, file, ec);
            if (ec) throw std::runtime_error("put_file: " + ec.message());
            d.uploads.erase(key);
            reply.result["path"] = file.generic_u8string();
            d.log("received " + file.filename().u8string() + " (" + std::to_string(size) + " bytes)");
        }
        return reply;
    }

    rpc::Reply EngineNode::releaseOutputs(const rpc::Request& req) {
        int released = 0;
        if (req.params.contains("handles") && req.params["handles"].is_array())
            for (const json& h : req.params["handles"]) {
                std::string s, fp;
                std::uint64_t step = 0;
                if (!h.is_string() || !parseOutputHandle(h.get<std::string>(), s, step, fp) || s != impl_->session) continue;
                const std::optional<Executor::Held> held = impl_->executor->held(step);
                if (held && held->fingerprint == fp) {
                    impl_->executor->invalidate(step);
                    ++released;
                }
            }
        rpc::Reply reply;
        reply.result = {{"released", released}};
        return reply;
    }

} // namespace sirius::app
