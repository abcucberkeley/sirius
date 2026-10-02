#ifndef SIRIUS_APP_DATASET_SERVICE_HPP
#define SIRIUS_APP_DATASET_SERVICE_HPP

// Datasets read where they live, served in C++: dataset_info / dataset_read /
// dataset_view / dataset_stats of the worker protocol, as
// app/python/sirius_worker/datasets.py answers them -- the same replies, so
// the application's RemoteSource (core/remote_source.hpp) cannot tell which
// one it talks to. The engine on a cluster node (core/engine_server.hpp)
// serves its datasets with this.
//
//   meta    dims (c, t, z, y, x), dtype, voxel size, channels
//   plane   one (y, x) plane at full resolution, in the file's pixel type
//   volume  one (z, y, x) volume at full resolution
//   view    what a pane draws: the XY plane at z, the XZ / YZ re-slice at y /
//           x or the z maximum projection, of a region, reduced by an integer
//           factor (the mean of each factor x factor block in the source
//           dtype: integers rounded), or the volume reduced to a longest side
//   stats   a display window: the 0.1 / 99.9 percentiles of a few planes and the range
//
// Readers: TIFF / OME-TIFF / ImageJ hyperstacks through SIRIUS's own TIFF
// reader, shaped exactly as openDataset shapes them (probeTiffDataset), with
// nvTIFF on a CUDA device when the build has it and the request's "device"
// asks for one; and .npy arrays. What is read is what is needed: a z-stack is
// one page range, a zoomed-in XY plane only its region, a reduced XY plane a
// pyramid level when the file has one. The (c, t) volumes read most recently
// are kept, up to $SIRIUS_WORKER_VIEW_CACHE_MB (default 4096), and the eight
// datasets opened last stay open while their files are unchanged.

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/rpc_server.hpp"

namespace sirius::app {

    // A dataset that cannot be opened or read as asked; the message is the
    // user's (sent as "DatasetError: <message>", as the Python worker does).
    class DatasetError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    // An array in a file's own pixel type, C order: dtype is rpc's name
    // ("uint16", ... and "bool", one byte each).
    struct HostArray {
        std::string dtype = "float32";
        std::vector<Index> shape;
        std::vector<std::byte> bytes;
        std::size_t elements() const noexcept;
    };

    // datasets.py reduce_blocks: the mean of every block of `factors` (one per
    // axis; a partial block at the end of an axis is the mean of what it has),
    // in the array's dtype: integers are rounded half to even and clipped,
    // float32 stays float32 (summed in float64), float64 float64. Factors of 1
    // everywhere return a copy. Integer means equal numpy's whenever a block's
    // sum is below 2^24 (numpy sums integers in float32).
    HostArray reduceBlocks(const HostArray& a, const std::vector<int>& factors);

    class DatasetService {
    public:
        struct Options {
            // where TIFF pages are decoded when a request names no device:
            // "auto" (the GPU when nvTIFF can decode there), "cpu", "cuda", "cuda:N"
            std::string device = "auto";
            // bytes of (c, t) volumes and projections kept; < 0: $SIRIUS_WORKER_VIEW_CACHE_MB, default 4096 MiB
            long long cacheBytes = -1;
        };

        DatasetService();
        explicit DatasetService(Options options);
        ~DatasetService();
        DatasetService(const DatasetService&) = delete;
        DatasetService& operator=(const DatasetService&) = delete;

        // dataset_info / dataset_read / dataset_view / dataset_stats
        static bool handles(const std::string& method);
        // One request; throws DatasetError (or std::exception) with the message to send back.
        rpc::Reply handle(const std::string& method, const nlohmann::json& params);

        // hello's "tiff_reader": {"sirius": this build's version, "nvtiff": a
        // decode on `device` (default: the service's) runs on the GPU}.
        nlohmann::json tiffReader(const std::string& device = {}) const;
        // Closes every dataset and drops the cached volumes.
        void forgetAll();
        // Bytes of volumes held now (tests, the engine's status).
        std::size_t cachedBytes() const;

        struct Impl;

    private:
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_DATASET_SERVICE_HPP
