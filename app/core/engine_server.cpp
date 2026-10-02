#include "core/engine_server.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <sirius/device.hpp>

#include "core/array_codec.hpp"
#include "core/build_info.hpp"
#include "core/cancel.hpp"
#include "core/errors.hpp"
#include "core/host.hpp"
#include "core/local_worker.hpp"

namespace sirius::app {

    using json = nlohmann::json;

    namespace {
        // What the engine serves itself.
        const std::vector<std::string>& ownMethods() {
            static const std::vector<std::string> m{"hello", "ping", "cancel", "shutdown",
                                                    "dataset_info", "dataset_read", "dataset_view", "dataset_stats"};
            return m;
        }
        // What it relays to the Python worker (server.py's method list, less
        // what is served here); run:<kind> come from the worker's own hello.
        const std::vector<std::string>& relayedMethods() {
            static const std::vector<std::string> m{"model_info", "run", "list_plugins", "reload_plugins", "hub_search", "hub_files",
                                                    "hub_download", "models_list", "models_delete", "install", "model_prepare", "list_bundles"};
            return m;
        }

        std::string lower(std::string s) {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        std::vector<rpc::TensorRef> refsOf(const std::vector<rpc::Tensor>& tensors) {
            std::vector<rpc::TensorRef> out;
            for (const rpc::Tensor& t : tensors) out.push_back(rpc::TensorRef{t.name, t.dtype, t.shape, t.bytes.data(), t.bytes.size()});
            return out;
        }
    } // namespace

    struct EngineServer::Impl {
        EngineOptions options;
        DatasetService datasets;

        // the Python child
        std::unique_ptr<LocalWorker> child;
        mutable std::mutex pythonMutex;
        std::string pythonState = "disabled", pythonError;
        json pythonCaps;
        std::vector<std::unique_ptr<RemoteWorker>> idle;   // connections to the child, ready for the next relay
        std::thread starter;

        // Last: destroyed first, so its connection and job threads (which use
        // everything above) have ended before any of it goes.
        rpc::Server server;

        explicit Impl(EngineOptions o)
            : options(std::move(o)), datasets(DatasetService::Options{options.device, options.viewCacheBytes}), server(serverOptions(options)) {}

        static rpc::Server::Options serverOptions(const EngineOptions& o) {
            rpc::Server::Options so;
            so.token = o.token;
            so.maxClients = o.maxClients;
            so.idleTimeout = o.idleTimeout;
            so.log = o.log;
            return so;
        }

        void log(const std::string& line) const {
            if (options.log) options.log(line);
        }

        bool pythonConfigured() const { return options.pythonWorker || static_cast<bool>(options.connectPython); }

        std::unique_ptr<RemoteWorker> connectChild(const std::function<bool()>& cancelled) {
            if (options.connectPython) return options.connectPython(cancelled);
            return child->connect(cancelled);
        }

        // A connection to the Python worker: an idle one, else a new one (which
        // starts the child when it is not running).
        std::unique_ptr<RemoteWorker> take(const std::function<bool()>& cancelled) {
            {
                const std::lock_guard<std::mutex> g(pythonMutex);
                while (!idle.empty()) {
                    std::unique_ptr<RemoteWorker> w = std::move(idle.back());
                    idle.pop_back();
                    if (w && w->isOpen()) return w;
                }
            }
            try {
                std::unique_ptr<RemoteWorker> w = connectChild(cancelled);
                if (!w) throw std::runtime_error("no connection");
                ready(w->capabilities());
                return w;
            } catch (const CancelledError&) {
                throw;
            } catch (const std::exception& e) {
                failed(e.what());
                throw std::runtime_error("the engine's Python worker is not available: " + std::string(e.what()));
            }
        }

        void giveBack(std::unique_ptr<RemoteWorker> w) {
            if (!w || !w->isOpen()) return;
            const std::lock_guard<std::mutex> g(pythonMutex);
            if (idle.size() < 4) idle.push_back(std::move(w));
        }

        void ready(const WorkerCapabilities& caps) {
            json c = {{"version", caps.version}, {"python", caps.python}, {"device", caps.device}, {"cuda", caps.cuda}, {"methods", caps.methods}};
            const std::lock_guard<std::mutex> g(pythonMutex);
            pythonState = "ready";
            pythonError.clear();
            pythonCaps = std::move(c);
        }

        void failed(const std::string& error) {
            const std::lock_guard<std::mutex> g(pythonMutex);
            pythonState = "failed";
            pythonError = error;
        }

        // Starts the child now, in the background: torch takes minutes to
        // import on a cluster's shared filesystem, and nothing waits for it.
        void startPython() {
            if (!pythonConfigured()) return;
            if (!options.connectPython) {
                child = std::make_unique<LocalWorker>();
                if (!options.python.empty()) child->setPython(options.python);
                if (!options.workerDir.empty()) child->setScriptDir(options.workerDir);
                child->setDevice(options.device.empty() ? std::string("auto") : options.device);
                child->setMaxClients(4);
                child->setLogHandler([this](const std::string& line) { log("python: " + line); });
            }
            {
                const std::lock_guard<std::mutex> g(pythonMutex);
                pythonState = "starting";
            }
            starter = std::thread([this] {
                try {
                    giveBack(take([this] { return server.stopping(); }));
                    log("the Python worker is ready");
                } catch (const std::exception& e) {
                    log(std::string("the Python worker did not start: ") + e.what());
                }
            });
        }

        rpc::Reply relay(const rpc::Request& req, rpc::CallContext& ctx) {
            if (!pythonConfigured())
                throw std::runtime_error("'" + req.method +
                                         "' is served by the engine's Python worker, and this engine runs without one (sirius-cli serve "
                                         "--no-python-worker); start it with a Python worker to use it");
            const std::function<bool()> cancelled = ctx.cancelFlag();
            std::unique_ptr<RemoteWorker> w = take(cancelled);
            try {
                WorkerResult r =
                    w->call(req.method, req.params, refsOf(req.tensors), [&ctx](double f, const std::string& m) { ctx.progress(f, m); }, cancelled);
                giveBack(std::move(w));
                rpc::Reply out;
                out.result = std::move(r.result);
                out.tensors = std::move(r.tensors);
                return out;
            } catch (const CancelledError&) {
                throw;
            } catch (const ProtocolError& e) {
                throw std::runtime_error(std::string("the engine's Python worker: ") + e.what());
            } catch (const std::runtime_error& e) {
                // the worker's own error, verbatim (RemoteWorker prefixes "worker: ")
                giveBack(std::move(w));
                std::string m = e.what();
                if (m.rfind("worker: ", 0) == 0) m = m.substr(8);
                throw std::runtime_error(m);
            }
        }

        std::string resolvedDevice() const {
            const std::string d = lower(options.device.empty() ? std::string("auto") : options.device);
            if (d == "cpu") return "cpu";
            if (d == "auto") return cudaAvailable() ? "cuda:0" : "cpu";
            if (d == "cuda") return "cuda:0";
            return d;
        }

        json pythonStatus() const {
            const std::lock_guard<std::mutex> g(pythonMutex);
            json s = {{"state", pythonState}};
            if (!pythonCaps.is_null()) s["caps"] = pythonCaps;
            if (!pythonError.empty()) s["error"] = pythonError;
            return s;
        }

        json capabilities() const {
            std::vector<std::string> methods = ownMethods();
            if (pythonConfigured()) {
                for (const std::string& m : relayedMethods()) methods.push_back(m);
                const std::lock_guard<std::mutex> g(pythonMutex);
                if (pythonCaps.contains("methods"))
                    for (const json& m : pythonCaps["methods"])
                        if (m.is_string() && std::find(methods.begin(), methods.end(), m.get<std::string>()) == methods.end())
                            methods.push_back(m.get<std::string>());
            }
            const std::string device = resolvedDevice();
            const int gpus = cudaDeviceCount();
            json devices = json::array();
            std::string deviceText = "cpu \xC2\xB7 " + std::to_string(std::max(1u, std::thread::hardware_concurrency())) + " threads";
            for (int i = 0; i < gpus; ++i) {
                try {
                    const DeviceProperties p = deviceProperties(Device::cuda(i));
                    const double gb = static_cast<double>(p.totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0);
                    devices.push_back({{"index", i}, {"name", p.name}, {"memory_gb", gb}});
                    if (device == "cuda:" + std::to_string(i))
                        deviceText = device + " \xC2\xB7 " + p.name + " \xC2\xB7 " + std::to_string(static_cast<int>(gb + 0.5)) + " GB";
                } catch (const std::exception&) {
                }
            }
            json engine = toJson(buildInfo());
            engine["cuda"] = {{"devices", devices}, {"nvtiff", builtWithNvTiff()}};
            engine["cpu_threads"] = std::max(1u, std::thread::hardware_concurrency());
            engine["view_cache_used"] = datasets.cachedBytes();
            engine["scratch"] = options.scratch;
            engine["job"] = {{"id", host::environment("SLURM_JOB_ID")}};
            engine["python"] = pythonStatus();
            std::string python;
            {
                const std::lock_guard<std::mutex> g(pythonMutex);
                if (pythonCaps.contains("python") && pythonCaps["python"].is_string()) python = pythonCaps["python"].get<std::string>();
            }
            return {{"version", buildInfo().version},
                    {"protocol_version", rpc::kProtocolVersion},
                    {"methods", methods},
                    {"cuda", gpus > 0 && device.rfind("cuda", 0) == 0},
                    {"device", deviceText},
                    {"hostname", host::hostName()},
                    {"python", python},
                    {"sirius", buildInfo().version},
                    {"encodings", codec::availableEncodings()},
                    {"max_clients", options.maxClients},
                    {"tiff_reader", datasets.tiffReader()},
                    {"engine", engine}};
        }
    };

    EngineServer::EngineServer(EngineOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {
        Impl& d = *impl_;
        for (const char* m : {"dataset_info", "dataset_read", "dataset_view", "dataset_stats"})
            d.server.handle(m, [this, m](const rpc::Request& req, rpc::CallContext&) { return impl_->datasets.handle(m, req.params); }, rpc::Dispatch::Inline);
        d.server.setFallback([this](const rpc::Request& req, rpc::CallContext& ctx) { return impl_->relay(req, ctx); }, rpc::Dispatch::Concurrent);
        d.server.setCapabilities([this] { return impl_->capabilities(); });
        d.startPython();
    }

    EngineServer::~EngineServer() {
        impl_->server.stop();
        if (impl_->child) impl_->child->stop();   // also ends a start in progress
        if (impl_->starter.joinable()) impl_->starter.join();
        {
            const std::lock_guard<std::mutex> g(impl_->pythonMutex);
            impl_->idle.clear();
        }
    }

    json EngineServer::capabilities() const { return impl_->capabilities(); }

    nlohmann::ordered_json EngineServer::announce(int port) const {
        const json engine = toJson(buildInfo());
        return {{"port", port},
                {"pid", host::processId()},
                {"host", impl_->options.host},
                {"hostname", host::hostName()},
                {"device", resolvedDevice()},
                {"engine", engine}};
    }

    std::string EngineServer::resolvedDevice() const { return impl_->resolvedDevice(); }
    json EngineServer::pythonStatus() const { return impl_->pythonStatus(); }

    void EngineServer::serveConnection(std::unique_ptr<rpc::Transport> transport, const std::string& peer) {
        impl_->server.serveConnection(std::move(transport), peer);
    }

    void EngineServer::serve(rpc::Listener& listener) { impl_->server.serve(listener); }
    void EngineServer::stop() { impl_->server.stop(); }
    bool EngineServer::stopping() const noexcept { return impl_->server.stopping(); }
    DatasetService& EngineServer::datasets() noexcept { return impl_->datasets; }
    rpc::Server& EngineServer::server() noexcept { return impl_->server; }

} // namespace sirius::app
