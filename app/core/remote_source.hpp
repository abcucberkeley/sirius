#ifndef SIRIUS_APP_REMOTE_SOURCE_HPP
#define SIRIUS_APP_REMOTE_SOURCE_HPP

// Datasets that stay on the cluster: "cluster://<ssh host>/<absolute path>",
// read by the worker on the node (app/python/sirius_worker/datasets.py)
// through the cluster session's SOCKS tunnel.
//
//   * The Load step opens one like a local file: openDataset() hands the path
//     to the opener RemoteDatasets installs, and gets a RemoteSource whose
//     meta came from dataset_info (cached per path and options).
//   * A step reads planes / volumes at full resolution (dataset_read), once
//     per (c, t), and a step that runs on the HPC worker is given the path
//     instead (inputReference(): the worker reads its input itself).
//   * The viewer draws through the ViewProvider (core/array_source.hpp): the
//     worker computes each pane's picture at the pane's resolution
//     (dataset_view), compressed on the wire (zstd or zlib with shuffled bytes,
//     core/array_codec.hpp, when the worker offers it); views arrive on a thread of the source's own and
//     are cached (bounded), with the neighbouring planes and time points
//     prefetched behind the visible ones.
//
// Each RemoteDatasets keeps two worker connections, one for views and one
// for full reads, so the dataset on screen is answered while a step reads.
// A connection that fails is dropped and made again on the next request.
//
// A step's output computed by SIRIUS's engine on the node stays there too
// (core/engine_server.hpp): its handle, "sirius-out:<engine session>/<step
// id>/<fingerprint>", takes the place of a path, and a NodeOutputSource draws
// it exactly as a cluster dataset is drawn.
//
// Whole volumes of what stays on the cluster are never read here by
// accident: RemoteSource::readVolume (and so readAll, StepInput::materialize)
// refuses unless the thread holds a RemoteDownloads::Allow naming why (a run
// the user chose to compute on this computer, an export the user agreed to
// download for). Planes (a probe, a contrast sample) are read as asked.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/array_source.hpp"
#include "core/rpc.hpp"

namespace sirius::app {

    // "cluster://fiona/home/u/a.tif" <-> ("fiona", "/home/u/a.tif").
    std::string makeClusterPath(const std::string& host, const std::string& remotePath);
    bool splitClusterPath(const std::string& path, std::string& host, std::string& remotePath);

    // A dataset_read / dataset_view reply's array as float32: the encoding
    // ("raw", "zlib", "zstd", shuffled bytes), the dtype, the shape. Throws
    // ProtocolError on a reply that does not add up.
    std::vector<float> decodeWorkerArray(const nlohmann::json& desc, const rpc::Tensor& data, std::vector<Index>& shape);

    // Bytes and time of what came over the wire, for the measurements.
    struct TransferStats {
        std::uint64_t requests = 0, wireBytes = 0, rawBytes = 0;
        double seconds = 0.0;
    };

    class RemoteDatasets : public std::enable_shared_from_this<RemoteDatasets> {
    public:
        using Connect = std::function<std::unique_ptr<RemoteWorker>()>;
        enum class Lane { Views,
                          Reads };

        // `host`: the ssh host the paths name; `connect` makes a new worker
        // connection (cluster::Session::connectWorker).
        RemoteDatasets(std::string host, Connect connect);
        ~RemoteDatasets();

        const std::string& host() const noexcept { return host_; }
        // Encodings asked for; "" in a test turns compression off.
        void setAccept(std::vector<std::string> accept);
        // The "device" every request carries ("cuda", "cpu"): where the
        // worker decodes the dataset's pages (nvTIFF on the GPU). The HPC
        // device of the session (Workbench::hpcDevice); empty sends none,
        // and the worker uses its own. Any thread.
        void setDevice(std::string device);
        std::string device() const;

        // One request on a lane's connection, made when there is none.
        WorkerResult call(Lane lane, const std::string& method, const nlohmann::json& params,
                          const std::function<bool()>& cancelled = {});
        // dataset_info as DatasetMeta (cached per path and options).
        DatasetMeta info(const std::string& remotePath, const OpenOptions& options);
        OpenResult open(const std::string& clusterPathName, const OpenOptions& options, bool probeOnly);

        // openDataset / probeDataset hand this object every cluster:// path of
        // its host from now on; uninstall() (or the destructor) stops that.
        void install();
        void uninstall();

        TransferStats stats() const;
        void resetStats();
        void account(std::uint64_t wire, std::uint64_t raw, double seconds);

    private:
        struct LaneState {
            std::mutex m;
            std::unique_ptr<RemoteWorker> worker;
        };
        std::string host_;
        Connect connect_;
        std::vector<std::string> accept_{"zstd", "zlib"};
        mutable std::mutex deviceMutex_;
        std::string device_;
        LaneState views_, reads_;
        std::mutex infoMutex_;
        std::map<std::string, DatasetMeta> infos_;
        // a path the worker could not open: asked again only after a while (the ops row asks every frame)
        std::map<std::string, std::pair<std::chrono::steady_clock::time_point, std::string>> infoErrors_;
        mutable std::mutex statsMutex_;
        TransferStats stats_;
        std::atomic<bool> installed_{false};
    };

    // The page order and the axes of the Load step's options, as the worker takes them.
    nlohmann::json remoteOptionsJson(const OpenOptions& options);

    // --- whole volumes of what stays on the cluster ----------------------------------------

    // Raised by a whole-volume read nobody allowed: what it would download and why it did not.
    class RemoteDataError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    class RemoteDownloads {
    public:
        // While one is alive, this thread may read whole volumes of data that
        // stays on the cluster; `purpose` goes into the log line each read
        // writes ("a run on this computer", "export"). Nested ones keep the outer purpose.
        class Allow {
        public:
            explicit Allow(std::string purpose);
            ~Allow();
            Allow(const Allow&) = delete;
            Allow& operator=(const Allow&) = delete;

        private:
            std::string previous_;
            bool had_ = false;
        };
        static bool allowed();
        static std::string purpose();
        // Bytes (float32, as decoded here) read so far by every thread: whole
        // volumes, and single planes. For the tests and the measurements.
        static std::uint64_t volumeBytes();
        static std::uint64_t planeBytes();
        // Called with one line per whole volume read ("downloading 3.2 GB of
        // <name> for export"); any thread. The application logs it.
        static void setObserver(std::function<void(const std::string&)> observer);
    };

    // --- outputs held by the engine on a node -------------------------------------------------

    // "sirius-out:<session>/<step id>/<fingerprint>": a step's output as the
    // engine holds it. The session is the engine process's (a new job, a new
    // session); the fingerprint is the node's own.
    inline constexpr const char* kOutputHandleScheme = "sirius-out:";
    std::string makeOutputHandle(const std::string& session, std::uint64_t step, const std::string& fingerprint);
    bool parseOutputHandle(const std::string& handle, std::string& session, std::uint64_t& step, std::string& fingerprint);
    bool isOutputHandle(const std::string& path);

    class RemoteSource : public ArraySource, public ViewProvider {
    public:
        RemoteSource(std::shared_ptr<RemoteDatasets> datasets, std::string remotePath, nlohmann::json options, DatasetMeta meta);
        ~RemoteSource() override;

        const DatasetMeta& meta() const noexcept override { return meta_; }
        void readPlane(Index c, Index t, Index z, float* out) const override;
        void readVolume(Index c, Index t, float* out, const ProgressFn& progress = {}) const override;
        ViewProvider* viewProvider() const noexcept override { return const_cast<RemoteSource*>(this); }

        // Channel statistics (core/statistics.hpp) computed where the data is:
        // the engine's output_stats, `request` and the reply as
        // statisticsOptionsToJson / channelStatisticsFromJson write them.
        // Throws when the peer is the Python worker, which has no such method.
        nlohmann::json statistics(const nlohmann::json& request, const std::function<bool()>& cancelled = {}) const;

        // The data went away with what held it (the cluster job ended): every
        // view and read from now on fails with `reason`, which lastError() says.
        void markGone(const std::string& reason);
        std::string gone() const;

        // ViewProvider
        std::shared_ptr<const ViewTile> view(const ViewRequest& request, bool& exact) override;
        std::optional<std::pair<float, float>> window(Index c, Index t, bool fullRange) override;
        std::uint64_t revision() const noexcept override { return revision_.load(); }
        bool busy() const override;
        std::string lastError() const override;

        // What a step on the HPC worker sends instead of the volume:
        // {"path", "options", "c", "t"} (layout "zyx").
        nlohmann::json inputReference(Index c, Index t) const;
        const nlohmann::json& options() const noexcept { return options_; }
        const std::string& remotePath() const noexcept { return path_; }
        const std::shared_ptr<RemoteDatasets>& datasets() const noexcept { return datasets_; }

        // Blocks until every queued view has been fetched (tests, measurements).
        void waitIdle(std::chrono::milliseconds timeout = std::chrono::seconds(30));
        // The tile a request would be served from, aligned for reuse (exposed for tests).
        static ViewRequest fetchRegion(const ViewRequest& r, const Dims5& dims);

    private:
        struct Key {
            int kind;
            Index c, t, index;
            bool operator<(const Key& o) const noexcept;
        };
        struct Pending {
            ViewRequest request;
            bool prefetch = false;
        };
        static Key keyOf(const ViewRequest& r);
        std::pair<int, int> viewSize(ViewRequest::Kind kind) const;   // (columns, rows) in view pixels
        void enqueue(const ViewRequest& r, bool prefetch);
        void fetchLoop();
        void fetchOne(const ViewRequest& r);
        nlohmann::json baseParams() const;

        std::shared_ptr<RemoteDatasets> datasets_;
        std::string path_;
        nlohmann::json options_;
        DatasetMeta meta_;

        mutable std::mutex m_;
        std::condition_variable wake_, idle_;
        std::map<Key, std::vector<std::shared_ptr<const ViewTile>>> tiles_;
        std::deque<std::pair<Key, std::size_t>> order_;   // tile insertion order, for the byte budget
        std::size_t bytes_ = 0;
        std::deque<Pending> queue_;
        std::vector<ViewRequest> inFlight_;
        std::map<std::pair<Index, Index>, std::array<float, 4>> windows_;   // (c, t) -> lo, hi, min, max
        std::map<std::pair<Index, Index>, bool> windowAsked_;
        std::map<std::string, std::chrono::steady_clock::time_point> failed_;
        std::string lastError_;
        std::string gone_;
        std::atomic<std::uint64_t> revision_{1};
        bool quit_ = false;
        bool busy_ = false;
        std::thread thread_;
        // full-resolution volumes read for steps: the last few (c, t)
        mutable std::mutex volMutex_;
        mutable std::deque<std::pair<std::pair<Index, Index>, std::shared_ptr<std::vector<float>>>> volumes_;
    };

    // A step's output that the engine on a node computed and holds, drawn at
    // screen size as a cluster dataset is (its handle is the path), never
    // downloaded unless asked for. Its meta came with the run's result.
    class NodeOutputSource final : public RemoteSource {
    public:
        // `datasets` connects to the engine; `where` says which job holds it
        // ("fiona · n0042 · job 4711").
        NodeOutputSource(std::shared_ptr<RemoteDatasets> datasets, std::string handle, DatasetMeta meta, std::string where);

        bool heldByNodeCache() const noexcept override { return true; }
        const std::string& handle() const noexcept { return remotePath(); }
        const std::string& session() const noexcept { return session_; }
        const std::string& where() const noexcept { return where_; }

    private:
        std::string session_, where_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_REMOTE_SOURCE_HPP
