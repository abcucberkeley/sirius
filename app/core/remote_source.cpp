#include "core/remote_source.hpp"

#include "core/errors.hpp"

#include <sirius/checked_math.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <zlib.h>

namespace sirius::app {

    using json = nlohmann::json;

    namespace {
        constexpr const char* kScheme = "cluster://";
        // Views kept on this side: what scrubbing back and forth reuses.
        constexpr std::size_t kViewBudget = std::size_t{512} << 20;
        // A reduced plane up to this many pixels a side is fetched whole: every
        // pan at that zoom is then local. Past it, aligned regions.
        constexpr int kWholePlaneSide = 2048;
        constexpr std::size_t kMaxPrefetch = 24;

        std::size_t dtypeBytes(const std::string& dtype) {
            if (dtype == "uint8" || dtype == "int8" || dtype == "bool") return 1;
            if (dtype == "uint16" || dtype == "int16") return 2;
            if (dtype == "uint32" || dtype == "int32" || dtype == "float32") return 4;
            if (dtype == "uint64" || dtype == "int64" || dtype == "float64") return 8;
            throw ProtocolError("worker: an array of unsupported dtype '" + dtype + "'");
        }

        template <typename T> void widen(const std::byte* p, std::size_t n, float* out) {
            for (std::size_t i = 0; i < n; ++i) {
                T v;
                std::memcpy(&v, p + i * sizeof(T), sizeof(T));
                out[i] = static_cast<float>(v);
            }
        }

        std::vector<std::byte> inflateAll(const std::vector<std::byte>& in, std::size_t expected) {
            std::vector<std::byte> out(expected);
            z_stream zs{};
            if (inflateInit(&zs) != Z_OK) throw ProtocolError("worker: zlib cannot start");
            std::size_t inPos = 0, outPos = 0;
            int rc = Z_OK;
            constexpr std::size_t kChunk = std::size_t{1} << 30;   // zlib counts in uInt
            while (rc != Z_STREAM_END) {
                const std::size_t inLeft = in.size() - inPos, outLeft = out.size() - outPos;
                zs.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(in.data() + inPos));
                zs.avail_in = static_cast<uInt>(std::min(inLeft, kChunk));
                zs.next_out = reinterpret_cast<Bytef*>(out.data() + outPos);
                zs.avail_out = static_cast<uInt>(std::min(outLeft, kChunk));
                const uInt availIn = zs.avail_in, availOut = zs.avail_out;
                rc = inflate(&zs, Z_NO_FLUSH);
                inPos += availIn - zs.avail_in;
                outPos += availOut - zs.avail_out;
                if (rc == Z_STREAM_END) break;
                if (rc != Z_OK || (availIn == zs.avail_in && availOut == zs.avail_out)) {
                    inflateEnd(&zs);
                    throw ProtocolError("worker: a compressed array does not decompress");
                }
            }
            inflateEnd(&zs);
            if (outPos != expected) throw ProtocolError("worker: a compressed array has the wrong size");
            return out;
        }

        std::string requestId(const ViewRequest& r) {
            return std::to_string(static_cast<int>(r.kind)) + ":" + std::to_string(r.c) + ":" + std::to_string(r.t) + ":" +
                   std::to_string(r.index) + ":" + std::to_string(r.factor) + ":" + std::to_string(r.x) + "," + std::to_string(r.y) + "," +
                   std::to_string(r.w) + "," + std::to_string(r.h);
        }

        bool sameRequest(const ViewRequest& a, const ViewRequest& b) { return requestId(a) == requestId(b); }

        PixelType pixelTypeOf(const std::string& dtype) {
            if (dtype == "uint8") return PixelType::UInt8;
            if (dtype == "int8") return PixelType::Int8;
            if (dtype == "uint16") return PixelType::UInt16;
            if (dtype == "int16") return PixelType::Int16;
            if (dtype == "uint32") return PixelType::UInt32;
            if (dtype == "int32") return PixelType::Int32;
            if (dtype == "float64") return PixelType::Float64;
            return PixelType::Float32;
        }
    } // namespace

    // --- names -------------------------------------------------------------------------

    std::string makeClusterPath(const std::string& host, const std::string& remotePath) {
        return std::string(kScheme) + host + (remotePath.rfind('/', 0) == 0 ? remotePath : "/" + remotePath);
    }

    bool splitClusterPath(const std::string& path, std::string& host, std::string& remotePath) {
        if (path.rfind(kScheme, 0) != 0) return false;
        const std::string rest = path.substr(std::strlen(kScheme));
        const std::size_t slash = rest.find('/');
        if (slash == std::string::npos || slash == 0) return false;
        host = rest.substr(0, slash);
        remotePath = rest.substr(slash);
        if (remotePath.size() > 2 && remotePath[2] == ':') remotePath.erase(0, 1);   // "/C:/x" -> "C:/x": a stand-in cluster on Windows
        return true;
    }

    // --- the wire form -------------------------------------------------------------------

    std::vector<float> decodeWorkerArray(const json& desc, const rpc::Tensor& data, std::vector<Index>& shape) {
        shape.clear();
        if (!desc.contains("shape") || !desc["shape"].is_array()) throw ProtocolError("worker: an array without a shape");
        if (desc["shape"].size() > 8) throw ProtocolError("worker: an array of " + std::to_string(desc["shape"].size()) + " dimensions");
        for (const json& d : desc["shape"]) {
            if (!d.is_number_integer() || d.get<std::int64_t>() < 0) throw ProtocolError("worker: a malformed array shape");
            shape.push_back(d.get<Index>());
        }
        const std::string dtype = desc.value("dtype", std::string("float32"));
        const std::size_t item = dtypeBytes(dtype);
        // The shape comes from the worker: its product is checked before it
        // sizes the output, and a compressed array may claim only what its
        // bytes can hold (zlib inflates at most ~1032:1) -- a description
        // that says more is refused before anything is allocated for it.
        std::size_t n = 0, expected = 0;
        try {
            n = static_cast<std::size_t>(sirius::detail::checkedProduct(shape.begin(), shape.end(), "worker: array shape"));
            expected = sirius::detail::checkedBytes(static_cast<std::ptrdiff_t>(n), item, "worker: array size");
            (void)sirius::detail::checkedBytes(static_cast<std::ptrdiff_t>(n), sizeof(float), "worker: array size");
        } catch (const std::exception& e) {
            throw ProtocolError(e.what());
        }
        if (expected > rpc::maxPayloadBytes()) throw ProtocolError("worker: an array of " + std::to_string(expected) + " bytes exceeds the limit");
        const std::string encoding = desc.value("encoding", std::string("raw"));
        std::vector<std::byte> raw;
        const std::vector<std::byte>* bytes = &data.bytes;
        if (encoding == "zlib") {
            if (expected / 1100 > data.bytes.size() + 1)
                throw ProtocolError("worker: a compressed array of " + std::to_string(data.bytes.size()) + " bytes claims " + std::to_string(expected));
            raw = inflateAll(data.bytes, expected);
            if (desc.value("shuffle", false) && item > 1) {
                std::vector<std::byte> un(raw.size());
                for (std::size_t b = 0; b < item; ++b)
                    for (std::size_t i = 0; i < n; ++i) un[i * item + b] = raw[b * n + i];
                raw.swap(un);
            }
            bytes = &raw;
        } else if (encoding != "raw") {
            throw ProtocolError("worker: an array encoded as '" + encoding + "', which this application does not read");
        }
        if (bytes->size() != n * item) throw ProtocolError("worker: an array's bytes do not match its shape");
        std::vector<float> out(n);
        const std::byte* p = bytes->data();
        if (dtype == "uint8" || dtype == "bool") widen<std::uint8_t>(p, n, out.data());
        else if (dtype == "int8") widen<std::int8_t>(p, n, out.data());
        else if (dtype == "uint16") widen<std::uint16_t>(p, n, out.data());
        else if (dtype == "int16") widen<std::int16_t>(p, n, out.data());
        else if (dtype == "uint32") widen<std::uint32_t>(p, n, out.data());
        else if (dtype == "int32") widen<std::int32_t>(p, n, out.data());
        else if (dtype == "uint64") widen<std::uint64_t>(p, n, out.data());
        else if (dtype == "int64") widen<std::int64_t>(p, n, out.data());
        else if (dtype == "float64") widen<double>(p, n, out.data());
        else std::memcpy(out.data(), p, n * sizeof(float));
        return out;
    }

    json remoteOptionsJson(const OpenOptions& o) {
        json j = json::object();
        if (o.pageOrder) {
            j["page_order"] = o.pageOrder->order;
            j["c"] = o.pageOrder->c;
            j["t"] = o.pageOrder->t;
            j["z"] = o.pageOrder->z;
        }
        return j;
    }

    // --- the datasets of one session ---------------------------------------------------------

    RemoteDatasets::RemoteDatasets(std::string host, Connect connect) : host_(std::move(host)), connect_(std::move(connect)) {
        // zstd needs a decoder this build does not have: zlib, which the worker always offers
        accept_ = {"zlib"};
    }

    RemoteDatasets::~RemoteDatasets() { uninstall(); }

    void RemoteDatasets::setAccept(std::vector<std::string> accept) { accept_ = std::move(accept); }

    WorkerResult RemoteDatasets::call(Lane lane, const std::string& method, const json& paramsIn, const std::function<bool()>& cancelled) {
        LaneState& l = lane == Lane::Views ? views_ : reads_;
        const std::lock_guard<std::mutex> g(l.m);
        json params = paramsIn;
        if (!params.contains("accept")) params["accept"] = accept_;
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (!l.worker || !l.worker->isOpen()) {
                l.worker.reset();
                l.worker = connect_();
                if (!l.worker) throw ProtocolError("no connection to the cluster's worker");
                l.worker->setCancelGrace(std::chrono::milliseconds(0));
            }
            try {
                const auto t0 = std::chrono::steady_clock::now();
                WorkerResult r = l.worker->call(method, params, {}, {}, cancelled);
                std::uint64_t wire = 0;
                for (const rpc::Tensor& t : r.tensors) wire += t.bytes.size();
                const std::uint64_t rawBytes = r.result.value("raw_bytes", static_cast<std::uint64_t>(wire));
                account(wire, rawBytes, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                return r;
            } catch (const ProtocolError&) {
                // the connection broke (the worker restarted, the tunnel hiccuped): once more on a new one
                l.worker.reset();
                if (attempt == 1) throw;
            } catch (...) {
                // a cancelled or failed request leaves the connection in an unknown state
                if (!l.worker || !l.worker->isOpen()) l.worker.reset();
                throw;
            }
        }
        throw ProtocolError("unreachable");
    }

    DatasetMeta RemoteDatasets::info(const std::string& remotePath, const OpenOptions& options) {
        const json opts = remoteOptionsJson(options);
        const std::string key = remotePath + "\n" + opts.dump();
        {
            const std::lock_guard<std::mutex> g(infoMutex_);
            auto it = infos_.find(key);
            if (it != infos_.end()) return it->second;
            auto err = infoErrors_.find(key);
            if (err != infoErrors_.end() && std::chrono::steady_clock::now() - err->second.first < std::chrono::seconds(10))
                throw std::runtime_error(err->second.second);
        }
        WorkerResult r;
        try {
            r = call(Lane::Reads, "dataset_info", {{"path", remotePath}, {"options", opts}});
        } catch (const std::exception& e) {
            const std::lock_guard<std::mutex> g(infoMutex_);
            infoErrors_[key] = {std::chrono::steady_clock::now(), e.what()};
            throw;
        }
        const json& j = r.result;
        DatasetMeta m;
        m.name = j.value("name", std::string());
        m.sourcePath = makeClusterPath(host_, j.value("path", remotePath));
        m.format = "cluster " + j.value("format", std::string("tiff"));
        if (!j.contains("dims") || !j["dims"].is_array() || j["dims"].size() != 5) throw ProtocolError("worker: dataset_info without dims");
        m.dims = Dims5{j["dims"][0].get<Index>(), j["dims"][1].get<Index>(), j["dims"][2].get<Index>(), j["dims"][3].get<Index>(),
                       j["dims"][4].get<Index>()};
        m.sourceType = pixelTypeOf(j.value("dtype", std::string("float32")));
        m.bytesOnDisk = j.value("bytes", static_cast<std::uint64_t>(0));
        if (j.contains("voxel_um") && j["voxel_um"].is_array() && j["voxel_um"].size() == 3)
            for (std::size_t k = 0; k < 3; ++k) {
                const double v = j["voxel_um"][k].is_number() ? j["voxel_um"][k].get<double>() : 0.0;
                if (v > 0.0) m.voxelUm[k] = v;
            }
        if (options.voxelUm)
            for (std::size_t k = 0; k < 3; ++k)
                if ((*options.voxelUm)[k] > 0.0) m.voxelUm[k] = (*options.voxelUm)[k];
        m.frameIntervalS = j.value("frame_interval_s", 0.0);
        m.rgb = j.value("rgb", false);
        if (j.contains("channels") && j["channels"].is_array())
            for (const json& c : j["channels"]) {
                ChannelInfo ci;
                ci.label = c.value("name", std::string());
                ci.wavelengthNm = c.value("wavelength_nm", 0.0);
                if (ci.wavelengthNm > 0.0) ci.color = colorForWavelength(ci.wavelengthNm);
                m.channels.push_back(ci);
            }
        if (options.channels) m.channels = *options.channels;
        if (options.sim) m.sim = *options.sim;
        m.normalizeChannels();
        const std::lock_guard<std::mutex> g(infoMutex_);
        infos_[key] = m;
        return m;
    }

    OpenResult RemoteDatasets::open(const std::string& name, const OpenOptions& options, bool probeOnly) {
        std::string host, remotePath;
        if (!splitClusterPath(name, host, remotePath)) throw std::runtime_error("not a cluster path: " + name);
        if (host != host_)
            throw std::runtime_error(name + " is on " + host + ", but the session is connected to " + host_ +
                                     ": connect to " + host + " first (Process \xE2\x96\xB8 Connect to cluster\xE2\x80\xA6)");
        OpenResult r;
        r.meta = info(remotePath, options);
        r.dimsFromMetadata = true;
        r.metadataSummary = "on " + host_ + " \xC2\xB7 " + r.meta.format.substr(8) + " \xC2\xB7 " + r.meta.shapeString();
        if (options.readAll)
            r.fullLoadSkipped = "a cluster dataset stays on the cluster: the viewer gets what it draws, a step on the HPC "
                                "backend reads it on the node";
        if (!probeOnly) r.source = std::make_shared<RemoteSource>(shared_from_this(), remotePath, remoteOptionsJson(options), r.meta);
        return r;
    }

    void RemoteDatasets::install() {
        std::weak_ptr<RemoteDatasets> weak = shared_from_this();
        setRemoteDatasetOpener([weak](const std::string& path, const OpenOptions& options, bool probeOnly) {
            auto self = weak.lock();
            if (!self) throw std::runtime_error("not connected to the cluster this dataset is on (" + path + ")");
            return self->open(path, options, probeOnly);
        });
        installed_.store(true);
    }

    void RemoteDatasets::uninstall() {
        if (installed_.exchange(false)) setRemoteDatasetOpener({});
    }

    TransferStats RemoteDatasets::stats() const {
        const std::lock_guard<std::mutex> g(statsMutex_);
        return stats_;
    }

    void RemoteDatasets::resetStats() {
        const std::lock_guard<std::mutex> g(statsMutex_);
        stats_ = TransferStats{};
    }

    void RemoteDatasets::account(std::uint64_t wire, std::uint64_t raw, double seconds) {
        const std::lock_guard<std::mutex> g(statsMutex_);
        ++stats_.requests;
        stats_.wireBytes += wire;
        stats_.rawBytes += raw;
        stats_.seconds += seconds;
    }

    // --- the source ----------------------------------------------------------------------

    bool RemoteSource::Key::operator<(const Key& o) const noexcept {
        if (kind != o.kind) return kind < o.kind;
        if (c != o.c) return c < o.c;
        if (t != o.t) return t < o.t;
        return index < o.index;
    }

    RemoteSource::RemoteSource(std::shared_ptr<RemoteDatasets> datasets, std::string remotePath, json options, DatasetMeta meta)
        : datasets_(std::move(datasets)), path_(std::move(remotePath)), options_(std::move(options)), meta_(std::move(meta)) {
        thread_ = std::thread([this] { fetchLoop(); });
    }

    RemoteSource::~RemoteSource() {
        {
            const std::lock_guard<std::mutex> g(m_);
            quit_ = true;
        }
        wake_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    json RemoteSource::baseParams() const { return {{"path", path_}, {"options", options_}}; }

    json RemoteSource::inputReference(Index c, Index t) const {
        return {{"path", path_}, {"options", options_}, {"c", c}, {"t", t}, {"layout", "zyx"}};
    }

    void RemoteSource::readPlane(Index c, Index t, Index z, float* out) const {
        const Dims5& d = meta_.dims;
        const std::size_t plane = static_cast<std::size_t>(d.y * d.x);
        {
            const std::lock_guard<std::mutex> g(volMutex_);
            for (const auto& [key, vol] : volumes_)
                if (key == std::make_pair(c, t)) {
                    std::memcpy(out, vol->data() + static_cast<std::size_t>(z) * plane, plane * sizeof(float));
                    return;
                }
        }
        json p = baseParams();
        p["c"] = c;
        p["t"] = t;
        p["z"] = z;
        const WorkerResult r = datasets_->call(RemoteDatasets::Lane::Reads, "dataset_read", p);
        if (r.tensors.empty()) throw ProtocolError("worker: dataset_read sent no array");
        std::vector<Index> shape;
        const std::vector<float> v = decodeWorkerArray(r.result, r.tensors.front(), shape);
        if (v.size() != plane) throw ProtocolError("worker: the plane does not match the dataset's size");
        std::memcpy(out, v.data(), plane * sizeof(float));
    }

    void RemoteSource::readVolume(Index c, Index t, float* out, const ProgressFn& progress) const {
        const Dims5& d = meta_.dims;
        const std::size_t n = static_cast<std::size_t>(d.z * d.y * d.x);
        {
            const std::lock_guard<std::mutex> g(volMutex_);
            for (const auto& [key, vol] : volumes_)
                if (key == std::make_pair(c, t)) {
                    std::memcpy(out, vol->data(), n * sizeof(float));
                    return;
                }
        }
        if (progress) progress(0.0, "reading c " + std::to_string(c) + " t " + std::to_string(t) + " from " + datasets_->host());
        json p = baseParams();
        p["c"] = c;
        p["t"] = t;
        const WorkerResult r = datasets_->call(RemoteDatasets::Lane::Reads, "dataset_read", p);
        if (r.tensors.empty()) throw ProtocolError("worker: dataset_read sent no array");
        std::vector<Index> shape;
        auto v = std::make_shared<std::vector<float>>(decodeWorkerArray(r.result, r.tensors.front(), shape));
        if (v->size() != n) throw ProtocolError("worker: the volume does not match the dataset's size");
        std::memcpy(out, v->data(), n * sizeof(float));
        if (progress) progress(1.0, "");
        const std::lock_guard<std::mutex> g(volMutex_);
        volumes_.emplace_back(std::make_pair(c, t), std::move(v));
        // the last two (c, t), at most 2 GiB
        std::size_t held = 0;
        for (const auto& kv : volumes_) held += kv.second->size() * sizeof(float);
        while (volumes_.size() > 2 || (volumes_.size() > 1 && held > (std::size_t{2} << 30))) {
            held -= volumes_.front().second->size() * sizeof(float);
            volumes_.pop_front();
        }
    }

    // --- views ------------------------------------------------------------------------------

    RemoteSource::Key RemoteSource::keyOf(const ViewRequest& r) {
        return Key{static_cast<int>(r.kind), r.c, r.t, r.kind == ViewRequest::Kind::MIP || r.kind == ViewRequest::Kind::Volume ? 0 : r.index};
    }

    std::pair<int, int> RemoteSource::viewSize(ViewRequest::Kind kind) const {
        const Dims5& d = meta_.dims;
        switch (kind) {
            case ViewRequest::Kind::XZ: return {static_cast<int>(d.x), static_cast<int>(d.z)};
            case ViewRequest::Kind::YZ: return {static_cast<int>(d.z), static_cast<int>(d.y)};
            default: return {static_cast<int>(d.x), static_cast<int>(d.y)};
        }
    }

    ViewRequest RemoteSource::fetchRegion(const ViewRequest& r, const Dims5& d) {
        ViewRequest out = r;
        out.factor = std::max(r.factor, 1);
        if (r.kind == ViewRequest::Kind::Volume) {
            out.x = out.y = out.w = out.h = 0;
            out.factor = 1;
            return out;
        }
        int cols = static_cast<int>(d.x), rows = static_cast<int>(d.y);
        if (r.kind == ViewRequest::Kind::XZ) rows = static_cast<int>(d.z);
        if (r.kind == ViewRequest::Kind::YZ) cols = static_cast<int>(d.z);
        const int f = out.factor;
        if ((cols + f - 1) / f <= kWholePlaneSide && (rows + f - 1) / f <= kWholePlaneSide) {
            out.x = out.y = 0;
            out.w = cols;
            out.h = rows;
            return out;
        }
        const int align = f * 256;
        const int rx = r.w > 0 ? std::clamp(r.x, 0, cols - 1) : 0, ry = r.w > 0 ? std::clamp(r.y, 0, rows - 1) : 0;
        const int rw = r.w > 0 ? r.w : cols, rh = r.w > 0 ? r.h : rows;
        const int x0 = rx / align * align, y0 = ry / align * align;
        const int x1 = std::min(cols, (rx + rw + align - 1) / align * align), y1 = std::min(rows, (ry + rh + align - 1) / align * align);
        out.x = x0;
        out.y = y0;
        out.w = std::max(1, x1 - x0);
        out.h = std::max(1, y1 - y0);
        return out;
    }

    std::shared_ptr<const ViewTile> RemoteSource::view(const ViewRequest& req, bool& exact) {
        exact = false;
        const Key key = keyOf(req);
        const auto [cols, rows] = viewSize(req.kind);
        const int f = std::max(req.factor, 1);
        const int rx = req.w > 0 ? req.x : 0, ry = req.w > 0 ? req.y : 0;
        const int rw = req.w > 0 ? std::min(req.w, cols - rx) : cols, rh = req.w > 0 ? std::min(req.h, rows - ry) : rows;
        std::shared_ptr<const ViewTile> best;
        int bestScore = std::numeric_limits<int>::max();
        {
            const std::lock_guard<std::mutex> g(m_);
            auto it = tiles_.find(key);
            if (it != tiles_.end()) {
                for (const auto& tile : it->second) {
                    if (req.kind == ViewRequest::Kind::Volume) {
                        exact = true;
                        return tile;
                    }
                    const int tx1 = std::min(cols, tile->x + tile->w * tile->factor), ty1 = std::min(rows, tile->y + tile->h * tile->factor);
                    const bool covers = tile->x <= rx && tile->y <= ry && tx1 >= rx + rw && ty1 >= ry + rh;
                    if (tile->factor == f && covers) {
                        best = tile;
                        exact = true;
                        break;
                    }
                    const bool meets = tile->x < rx + rw && tx1 > rx && tile->y < ry + rh && ty1 > ry;
                    if (!meets) continue;
                    // the nearest factor, a finer one first; a covering one before a partial one
                    const int score = std::abs(tile->factor - f) * 4 + (tile->factor > f ? 1 : 0) + (covers ? 0 : 2);
                    if (score < bestScore) {
                        bestScore = score;
                        best = tile;
                    }
                }
            }
        }
        const ViewRequest fetch = fetchRegion(req, meta_.dims);
        if (!exact) enqueue(fetch, false);
        if (req.kind == ViewRequest::Kind::XY) {
            // the neighbours the user scrubs to next, behind the visible one
            for (Index dz : {1, -1, 2, -2}) {
                ViewRequest n = fetch;
                n.index = req.index + dz;
                if (n.index >= 0 && n.index < meta_.dims.z) enqueue(n, true);
            }
            for (Index dt : {1, -1}) {
                ViewRequest n = fetch;
                n.t = req.t + dt;
                if (n.t >= 0 && n.t < meta_.dims.t) enqueue(n, true);
            }
        } else if (req.kind == ViewRequest::Kind::MIP) {
            for (Index dt : {1, -1}) {
                ViewRequest n = fetch;
                n.t = req.t + dt;
                if (n.t >= 0 && n.t < meta_.dims.t) enqueue(n, true);
            }
        }
        return best;
    }

    void RemoteSource::enqueue(const ViewRequest& r, bool prefetch) {
        {
            const std::lock_guard<std::mutex> g(m_);
            const std::string id = requestId(r);
            auto fit = failed_.find(id);
            if (fit != failed_.end() && std::chrono::steady_clock::now() - fit->second < std::chrono::seconds(5)) return;
            // already here?
            auto it = tiles_.find(keyOf(r));
            if (it != tiles_.end())
                for (const auto& tile : it->second)
                    if (tile->factor == r.factor && tile->x <= r.x && tile->y <= r.y && tile->x + tile->w * tile->factor >= r.x + r.w &&
                        tile->y + tile->h * tile->factor >= r.y + r.h)
                        return;
            for (const ViewRequest& f : inFlight_)
                if (sameRequest(f, r)) return;
            for (auto q = queue_.begin(); q != queue_.end(); ++q)
                if (sameRequest(q->request, r)) {
                    if (prefetch || !q->prefetch) return;
                    queue_.erase(q);   // promoted: a prefetch the user now looks at
                    break;
                }
            if (prefetch) {
                queue_.push_back(Pending{r, true});
                std::size_t n = 0;
                for (const Pending& p : queue_) n += p.prefetch ? 1 : 0;
                if (n > kMaxPrefetch)
                    for (auto q = queue_.begin(); q != queue_.end(); ++q)
                        if (q->prefetch) {
                            queue_.erase(q);
                            break;
                        }
            } else {
                // what the pane showed a moment ago is no longer wanted
                queue_.erase(std::remove_if(queue_.begin(), queue_.end(),
                                            [&](const Pending& p) {
                                                return !p.prefetch && p.request.kind == r.kind && p.request.c == r.c &&
                                                       p.request.maxSide >= 0;
                                            }),
                             queue_.end());
                queue_.push_front(Pending{r, false});
            }
        }
        wake_.notify_all();
    }

    std::optional<std::pair<float, float>> RemoteSource::window(Index c, Index t, bool fullRange) {
        {
            const std::lock_guard<std::mutex> g(m_);
            auto it = windows_.find({c, t});
            if (it != windows_.end()) {
                const auto& w = it->second;
                return fullRange ? std::make_pair(w[2], w[3]) : std::make_pair(w[0], w[1]);
            }
            if (windowAsked_[{c, t}]) return std::nullopt;
            windowAsked_[{c, t}] = true;
            ViewRequest r;
            r.c = c;
            r.t = t;
            r.maxSide = -1;   // marks a window request
            queue_.push_front(Pending{r, false});
        }
        wake_.notify_all();
        return std::nullopt;
    }

    bool RemoteSource::busy() const {
        const std::lock_guard<std::mutex> g(m_);
        return busy_ || !queue_.empty();
    }

    std::string RemoteSource::lastError() const {
        const std::lock_guard<std::mutex> g(m_);
        return lastError_;
    }

    void RemoteSource::waitIdle(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(m_);
        idle_.wait_for(lk, timeout, [this] { return queue_.empty() && !busy_; });
    }

    void RemoteSource::fetchLoop() {
        for (;;) {
            Pending p;
            {
                std::unique_lock<std::mutex> lk(m_);
                busy_ = false;
                idle_.notify_all();
                wake_.wait(lk, [this] { return quit_ || !queue_.empty(); });
                if (quit_) return;
                p = queue_.front();
                queue_.pop_front();
                inFlight_.push_back(p.request);
                busy_ = true;
            }
            fetchOne(p.request);
            {
                const std::lock_guard<std::mutex> g(m_);
                inFlight_.erase(std::remove_if(inFlight_.begin(), inFlight_.end(), [&](const ViewRequest& r) { return sameRequest(r, p.request); }),
                                inFlight_.end());
            }
        }
    }

    void RemoteSource::fetchOne(const ViewRequest& r) {
        try {
            json params = baseParams();
            params["c"] = r.c;
            params["t"] = r.t;
            if (r.maxSide < 0) {
                const WorkerResult res = datasets_->call(RemoteDatasets::Lane::Views, "dataset_stats", params);
                const std::array<float, 4> w{res.result.value("lo", 0.0f), res.result.value("hi", 1.0f), res.result.value("min", 0.0f),
                                             res.result.value("max", 1.0f)};
                {
                    const std::lock_guard<std::mutex> g(m_);
                    windows_[{r.c, r.t}] = w;
                }
                revision_.fetch_add(1);
                return;
            }
            static const char* kinds[] = {"xy", "xz", "yz", "mip", "volume"};
            params["kind"] = kinds[static_cast<int>(r.kind)];
            params["index"] = r.index;
            params["factor"] = r.factor;
            if (r.kind != ViewRequest::Kind::Volume) params["region"] = {r.x, r.y, r.w, r.h};
            params["max_side"] = r.maxSide;
            const WorkerResult res = datasets_->call(RemoteDatasets::Lane::Views, "dataset_view", params);
            if (res.tensors.empty()) throw ProtocolError("worker: dataset_view sent no array");
            std::vector<Index> shape;
            auto tile = std::make_shared<ViewTile>();
            tile->data = decodeWorkerArray(res.result, res.tensors.front(), shape);
            tile->factor = r.factor;
            tile->x = r.x;
            tile->y = r.y;
            if (shape.size() == 3) {
                tile->d = static_cast<int>(shape[0]);
                tile->h = static_cast<int>(shape[1]);
                tile->w = static_cast<int>(shape[2]);
            } else if (shape.size() == 2) {
                tile->h = static_cast<int>(shape[0]);
                tile->w = static_cast<int>(shape[1]);
            } else {
                throw ProtocolError("worker: a view of rank " + std::to_string(shape.size()));
            }
            const Key key = keyOf(r);
            const std::size_t bytes = tile->data.size() * sizeof(float);
            {
                const std::lock_guard<std::mutex> g(m_);
                tiles_[key].push_back(tile);
                order_.emplace_back(key, bytes);
                bytes_ += bytes;
                while (bytes_ > kViewBudget && order_.size() > 1) {
                    const auto [oldKey, oldBytes] = order_.front();
                    order_.pop_front();
                    auto it = tiles_.find(oldKey);
                    if (it != tiles_.end() && !it->second.empty()) {
                        it->second.erase(it->second.begin());
                        if (it->second.empty()) tiles_.erase(it);
                    }
                    bytes_ -= oldBytes;
                }
                lastError_.clear();
            }
            revision_.fetch_add(1);
        } catch (const std::exception& e) {
            {
                const std::lock_guard<std::mutex> g(m_);
                lastError_ = e.what();
                failed_[requestId(r)] = std::chrono::steady_clock::now();
            }
            revision_.fetch_add(1);
        }
    }

} // namespace sirius::app
