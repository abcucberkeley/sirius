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
//     (dataset_view), compressed on the wire (zlib with shuffled bytes, when
//     the worker offers it); views arrive on a thread of the source's own and
//     are cached (bounded), with the neighbouring planes and time points
//     prefetched behind the visible ones.
//
// Each RemoteDatasets keeps two worker connections, one for views and one
// for full reads, so the dataset on screen is answered while a step reads.
// A connection that fails is dropped and made again on the next request.

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
    // ("raw", "zlib", shuffled bytes), the dtype, the shape. Throws
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

    class RemoteSource final : public ArraySource, public ViewProvider {
    public:
        RemoteSource(std::shared_ptr<RemoteDatasets> datasets, std::string remotePath, nlohmann::json options, DatasetMeta meta);
        ~RemoteSource() override;

        const DatasetMeta& meta() const noexcept override { return meta_; }
        void readPlane(Index c, Index t, Index z, float* out) const override;
        void readVolume(Index c, Index t, float* out, const ProgressFn& progress = {}) const override;
        ViewProvider* viewProvider() const noexcept override { return const_cast<RemoteSource*>(this); }

        // ViewProvider
        std::shared_ptr<const ViewTile> view(const ViewRequest& request, bool& exact) override;
        std::optional<std::pair<float, float>> window(Index c, Index t, bool fullRange) override;
        std::uint64_t revision() const noexcept override { return revision_.load(); }
        bool busy() const override;
        std::string lastError() const override;

        // What a step on the HPC worker sends instead of the volume:
        // {"path", "options", "c", "t"} (layout "zyx").
        nlohmann::json inputReference(Index c, Index t) const;
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
        std::atomic<std::uint64_t> revision_{1};
        bool quit_ = false;
        bool busy_ = false;
        std::thread thread_;
        // full-resolution volumes read for steps: the last few (c, t)
        mutable std::mutex volMutex_;
        mutable std::deque<std::pair<std::pair<Index, Index>, std::shared_ptr<std::vector<float>>>> volumes_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_REMOTE_SOURCE_HPP
