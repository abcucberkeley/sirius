#include "core/ops/plugin.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <cctype>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>

#include "core/array_source.hpp"
#include "core/help_pages.hpp"
#include "core/ops/builtin.hpp"

namespace sirius::app {

    using json = nlohmann::json;

    namespace {
        std::set<std::string>& registeredKinds() {
            static std::set<std::string> kinds;
            return kinds;
        }
        std::mutex& kindsMutex() {
            static std::mutex m;
            return m;
        }

        ParamSpec specFromJson(const json& p) {
            const std::string key = p.value("key", "");
            const std::string label = p.value("label", key);
            const std::string type = p.value("type", "double");
            ParamSpec s;
            if (type == "double") s = doubleParam(key, label, p.value("default", 0.0));
            else if (type == "int") s = intParam(key, label, p.value("default", std::int64_t{0}));
            else if (type == "bool") s = boolParam(key, label, p.value("default", false));
            else if (type == "choice") {
                std::vector<std::string> choices = p.value("choices", std::vector<std::string>{});
                s = choiceParam(key, label, choices, p.value("default", choices.empty() ? std::string() : choices.front()));
            } else if (type == "path") s = pathParam(key, label, p.value("default", std::string()));
            else if (type == "string") s = stringParam(key, label, p.value("default", std::string()));
            else if (type == "channel") s = channelParam(key, label, p.value("default", std::int64_t{0}));
            else if (type == "axes") s = axesParam(key, label, p.value("default", std::string("ctzyx")));
            else if (type == "double_list") s = doubleListParam(key, label, p.value("default", std::vector<double>{}));
            else if (type == "string_list") {
                s = stringParam(key, label, std::string());
                s.type = ParamType::StringList;
                s.defaultValue = p.value("default", std::vector<std::string>{});
            } else throw std::invalid_argument("parameter '" + key + "': unknown type '" + type + "'");
            if (p.contains("min") && p["min"].is_number()) s.min = p["min"].get<double>();
            if (p.contains("max") && p["max"].is_number()) s.max = p["max"].get<double>();
            if (p.contains("step") && p["step"].is_number()) s.step = p["step"].get<double>();
            if (p.contains("decimals") && p["decimals"].is_number()) s.decimals = p["decimals"].get<int>();
            s.unit = p.value("unit", "");
            s.help = p.value("help", "");
            s.advanced = p.value("advanced", false);
            s.fileFilter = p.value("filter", "");
            return s;
        }

        std::string upper(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            return s;
        }

        class PluginOperation final : public Operation {
        public:
            explicit PluginOperation(const json& spec) {
                info_.kind = spec.value("kind", "");
                if (info_.kind.empty()) throw std::invalid_argument("plugin spec without a kind");
                info_.name = spec.value("name", info_.kind);
                // Every user operation sits in its own "User" section of the
                // add menu; the spec's group only names the row's kind label.
                info_.group = "User";
                const std::string declared = spec.value("group", "");
                info_.kindLabel = upper(declared.empty() || declared == "Plugins" || declared == "User" ? std::string("user") : declared);
                info_.diagnostics = DiagnosticsKind::Generic;
                info_.defaultCache = cachePolicyFromString(spec.value("cache", "memory")).value_or(CachePolicy::Memory);
                info_.separableOverT = spec.value("separable_over_t", false);
                info_.producesLabels = spec.value("produces_labels", false);
                info_.needsLabels = spec.value("needs_labels", false);
                info_.remoteCapable = true;
                info_.plugin = true;
                info_.source = spec.value("file", "");
                info_.helpPage = info_.kind;
                if (spec.contains("params") && spec["params"].is_array())
                    for (const json& p : spec["params"]) info_.params.push_back(specFromJson(p));
                help_ = spec.value("help", "");
                if (help_.empty())
                    help_ = "# " + info_.name + "\n\nA user operation from `" + info_.source + "` (the file has no help text).\n";
            }

            const OpInfo& info() const noexcept override { return info_; }
            const std::string& help() const noexcept { return help_; }

            Validation validate(const ParamSet& params, const DatasetMeta& input) const override {
                Validation v = Operation::validate(params, input);
                if (info_.needsLabels) v.warnings.push_back("Needs the labels of its input (a segmentation step before it).");
                return v;
            }

            StepOutput run(const StepInput& input, const ParamSet& params, const StepContext& ctx) const override {
                if (!ctx.remote)
                    throw std::runtime_error("The plugin '" + info_.name +
                                             "' runs in the Python worker: start it from Preferences ▸ Worker or choose the HPC backend");
                if (!ctx.remote->supports("plugin"))
                    throw std::runtime_error("The connected worker does not run plugins (update app/python/sirius_worker)");
                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                ctx.report(0.0, "sending");
                const json request = {{"kind", "plugin"}, {"plugin", info_.kind}, {"params", params.toJson()}, {"meta", metaJson(meta)}};

                std::shared_ptr<Array5> outArray;
                std::shared_ptr<LabelVolume> outLabels;
                std::vector<DiagnosticImage> images;
                json diagnostics, metaOut;
                double seconds = 0.0;

                // Folds one worker reply into the output: the whole result, or
                // time point `tIndex` of `tCount` when running per t.
                auto receive = [&](const WorkerResult& r, Index tIndex, Index tCount) {
                    const rpc::Tensor* out = nullptr;
                    const rpc::Tensor* labels = nullptr;
                    for (const rpc::Tensor& t : r.tensors) {
                        if (t.name == "output") out = &t;
                        else if (t.name == "labels") labels = &t;
                    }
                    if (!out || out->shape.size() != 5) throw std::runtime_error("the plugin returned no (c, t, z, y, x) output");
                    const Dims5 od{out->shape[0], out->shape[1], out->shape[2], out->shape[3], out->shape[4]};
                    if (!outArray) {
                        Dims5 full = od;
                        if (tCount > 1) full.t = tCount;
                        outArray = std::make_shared<Array5>(full);
                    }
                    const Dims5& fd = outArray->dims();
                    if (od.c != fd.c || od.z != fd.z || od.y != fd.y || od.x != fd.x || (tCount > 1 ? od.t != 1 : od.t != fd.t))
                        throw std::runtime_error("the plugin's output shape changed between time points");
                    const float* src = out->asFloat32();
                    const Index vol = od.z * od.y * od.x;
                    for (Index c = 0; c < od.c; ++c)
                        for (Index t = 0; t < od.t; ++t)
                            std::memcpy(outArray->plane(c, tCount > 1 ? tIndex : t, 0), src + (c * od.t + t) * vol,
                                        static_cast<std::size_t>(vol) * sizeof(float));
                    if (labels && labels->shape.size() == 4) {
                        const Index lt = labels->shape[0], lz = labels->shape[1], ly = labels->shape[2], lx = labels->shape[3];
                        if (!outLabels) outLabels = std::make_shared<LabelVolume>(tCount > 1 ? tCount : lt, lz, ly, lx);
                        for (Index t = 0; t < lt; ++t)
                            std::memcpy(outLabels->volume(tCount > 1 ? tIndex : t), labels->asUInt32() + t * lz * ly * lx,
                                        static_cast<std::size_t>(lz * ly * lx) * sizeof(std::uint32_t));
                    }
                    diagnostics = r.result.value("diagnostics", json::object());
                    metaOut = r.result.value("meta", json::object());
                    seconds += r.seconds;
                    // diagnostic images travel as extra tensors named in the header
                    if (diagnostics.contains("images") && diagnostics["images"].is_array()) {
                        for (const json& im : diagnostics["images"]) {
                            const std::string tensor = im.value("tensor", "");
                            for (const rpc::Tensor& t : r.tensors)
                                if (t.name == tensor && t.shape.size() == 2) {
                                    DiagnosticImage img;
                                    img.title = im.value("title", tensor);
                                    img.meta = im.value("meta", "");
                                    img.logScale = im.value("log", false);
                                    img.rows = t.shape[0];
                                    img.cols = t.shape[1];
                                    img.values.assign(t.asFloat32(), t.asFloat32() + t.numel());
                                    images.push_back(std::move(img));
                                }
                        }
                        diagnostics.erase("images");
                    }
                };

                std::vector<rpc::TensorRef> refs;
                if (info_.separableOverT && d.t > 1) {
                    for (Index t = 0; t < d.t; ++t) {
                        ctx.throwIfCancelled();
                        Array5 slab(Dims5{d.c, 1, d.z, d.y, d.x});   // one time point: (c, 1, z, y, x)
                        for (Index c = 0; c < d.c; ++c) {
                            Buffer<float> v = input.readVolume(c, t);
                            std::memcpy(slab.plane(c, 0, 0), v.data(), v.bytes());
                        }
                        refs.clear();
                        refs.push_back({"input", "float32", {d.c, 1, d.z, d.y, d.x}, slab.data(), slab.bytes()});
                        std::vector<std::uint32_t> labelCopy;
                        if (info_.needsLabels && input.labels && !input.labels->empty()) {
                            const Index n = input.labels->volumeSize();
                            labelCopy.assign(input.labels->volume(t), input.labels->volume(t) + n);
                            refs.push_back({"labels", "uint32", {1, input.labels->z(), input.labels->y(), input.labels->x()}, labelCopy.data(), labelCopy.size() * sizeof(std::uint32_t)});
                        }
                        const double base = static_cast<double>(t) / d.t, span = 1.0 / d.t;
                        const WorkerResult r = ctx.remote->call("run", request, refs, [&](double f, const std::string& m) { ctx.report(base + span * f, m); }, [&] { return ctx.isCancelled(); });
                        receive(r, t, d.t);
                    }
                } else {
                    ArrayPtr arr = input.materialize([&](double f, const std::string& m) { ctx.report(0.2 * f, m); });
                    refs.push_back({"input", "float32", {d.c, d.t, d.z, d.y, d.x}, arr->data(), arr->bytes()});
                    if (info_.needsLabels && input.labels && !input.labels->empty())
                        refs.push_back({"labels", "uint32", {input.labels->t(), input.labels->z(), input.labels->y(), input.labels->x()}, input.labels->volume(0), static_cast<std::size_t>(input.labels->t() * input.labels->volumeSize()) * sizeof(std::uint32_t)});
                    const WorkerResult r = ctx.remote->call("run", request, refs, [&](double f, const std::string& m) { ctx.report(0.2 + 0.8 * f, m); }, [&] { return ctx.isCancelled(); });
                    receive(r, 0, 1);
                }

                StepOutput out;
                out.meta = meta;
                out.meta.dims = outArray->dims();
                out.meta.sourceType = PixelType::Float32;
                applyMetaOverrides(out.meta, metaOut);
                if (out.meta.dims.c != static_cast<Index>(out.meta.channels.size())) out.meta.normalizeChannels();
                out.array = outArray;
                if (outLabels) {
                    for (Index t = 0; t < outLabels->t(); ++t) outLabels->recomputeStats(t);
                    outLabels->applyFlags(LabelFlagRules{});
                    out.labels = outLabels;
                } else if (input.labels) {
                    out.labels = input.labels->clone();
                }
                out.diagnostics = toDiagnostics(diagnostics, images, out);
                out.ranOn = ctx.backend;
                out.seconds = seconds;
                const std::string& dev = ctx.remote->capabilities().device;
                out.note = "plugin · " + (dev.empty() ? std::string("worker") : dev);
                ctx.report(1.0, "");
                return out;
            }

        private:
            static json metaJson(const DatasetMeta& m) {
                json channels = json::array();
                for (const ChannelInfo& c : m.channels)
                    channels.push_back({{"label", c.label}, {"wavelength_nm", c.wavelengthNm}, {"color", c.hexColor()}});
                return {{"dims", {m.dims.c, m.dims.t, m.dims.z, m.dims.y, m.dims.x}},
                        {"voxel_um", m.voxelUm},
                        {"channels", channels},
                        {"rgb", m.rgb},
                        {"name", m.name},
                        {"acquisition", m.acquisition}};
            }

            static void applyMetaOverrides(DatasetMeta& m, const json& o) {
                if (!o.is_object()) return;
                if (o.contains("voxel_um") && o["voxel_um"].is_array() && o["voxel_um"].size() == 3)
                    for (std::size_t i = 0; i < 3; ++i) m.voxelUm[i] = o["voxel_um"][i].get<double>();
                if (o.contains("rgb") && o["rgb"].is_boolean()) m.rgb = o["rgb"].get<bool>();
                if (o.contains("channels") && o["channels"].is_array()) {
                    std::vector<ChannelInfo> chans;
                    for (const json& c : o["channels"]) {
                        ChannelInfo ch;
                        if (c.is_string()) ch.label = c.get<std::string>();
                        else if (c.is_object()) {
                            ch.label = c.value("label", "");
                            ch.wavelengthNm = c.value("wavelength_nm", 0.0);
                            if (c.contains("color") && c["color"].is_string()) {
                                try {
                                    ch.color = colorFromHex(c["color"].get<std::string>());
                                } catch (const std::exception&) {}
                            }
                        }
                        chans.push_back(ch);
                    }
                    if (static_cast<Index>(chans.size()) == m.dims.c) m.channels = chans;
                }
                if (o.contains("acquisition") && o["acquisition"].is_string()) m.acquisition = o["acquisition"].get<std::string>();
            }

            Diagnostics toDiagnostics(const json& j, const std::vector<DiagnosticImage>& images, const StepOutput& out) const {
                Diagnostics d;
                d.kind = DiagnosticsKind::Generic;
                d.summary = j.value("summary", info_.name);
                if (j.contains("facts") && j["facts"].is_object())
                    for (auto it = j["facts"].begin(); it != j["facts"].end(); ++it)
                        d.facts.push_back({it.key(), it.value().is_string() ? it.value().get<std::string>() : it.value().dump()});
                if (j.contains("warnings") && j["warnings"].is_array())
                    for (const json& w : j["warnings"]) d.warnings.push_back(w.is_string() ? w.get<std::string>() : w.dump());
                d.footer = j.value("footer", "");
                if (j.contains("table") && j["table"].is_object()) {
                    DiagnosticTable t;
                    t.caption = j["table"].value("caption", "");
                    t.header = j["table"].value("header", std::vector<std::string>{});
                    if (j["table"].contains("rows") && j["table"]["rows"].is_array())
                        for (const json& row : j["table"]["rows"]) {
                            std::vector<std::string> cells;
                            for (const json& c : row) cells.push_back(c.is_string() ? c.get<std::string>() : c.dump());
                            t.rows.push_back(cells);
                        }
                    d.table = t;
                }
                d.images = images;
                if (d.images.empty() && out.array) {
                    const Dims5& od = out.array->dims();
                    d.addImage(thumbnail(out.array->plane(0, 0, od.z / 2), od.y, od.x, 256, "Output · z " + std::to_string(od.z / 2)));
                }
                d.facts.push_back({"Output", out.meta.shapeString()});
                return d;
            }

            OpInfo info_;
            std::string help_;
        };
    } // namespace

    std::unique_ptr<Operation> makePluginOperation(const json& spec) { return std::make_unique<PluginOperation>(spec); }

    PluginLoadResult registerPluginOperations(RemoteWorker& worker, bool reload) {
        PluginLoadResult result;
        const WorkerResult r = worker.call(reload ? "reload_plugins" : "list_plugins", json::object());
        result.dirs = r.result.value("dirs", std::vector<std::string>{});
        if (!r.result.contains("plugins") || !r.result["plugins"].is_array()) return result;
        std::set<std::string> builtins;   // a name a built-in owns stays a built-in
        for (const Operation* op : allOperations())
            if (!op->info().plugin) builtins.insert(op->kind());
        for (const json& spec : r.result["plugins"]) {
            const std::string file = spec.value("file", "?");
            PluginLoadResult::Entry entry{spec.value("kind", ""), spec.value("name", ""), file, ""};
            if (spec.contains("error")) {
                const std::string err = spec["error"].get<std::string>();
                entry.error = err;
                result.entries.push_back(entry);
                result.errors.push_back(file + ": " + err.substr(0, err.find('\n')));
                continue;
            }
            result.entries.push_back(entry);
            try {
                auto op = makePluginOperation(spec);
                const std::string kind = op->kind();
                if (builtins.count(kind)) {
                    result.errors.push_back(file + ": kind '" + kind + "' is a built-in operation");
                    continue;
                }
                registerHelpPage(kind, static_cast<PluginOperation*>(op.get())->help());
                registerOperation(std::move(op));
                {
                    std::lock_guard<std::mutex> g(kindsMutex());
                    registeredKinds().insert(kind);
                }
                result.kinds.push_back(kind);
            } catch (const std::exception& e) {
                result.errors.push_back(file + ": " + e.what());
            }
        }
        return result;
    }

    std::string userPluginDirectory(bool create) {
        const char* home = std::getenv("HOME");
#ifdef _WIN32
        if (!home) home = std::getenv("USERPROFILE");
#endif
        const std::filesystem::path dir = std::filesystem::path(home ? home : ".") / ".sirius" / "plugins";
        if (create) {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
        }
        return dir.string();
    }

    std::vector<std::string> pluginKinds() {
        std::lock_guard<std::mutex> g(kindsMutex());
        return std::vector<std::string>(registeredKinds().begin(), registeredKinds().end());
    }

} // namespace sirius::app
