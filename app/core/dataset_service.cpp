#include "core/dataset_service.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <type_traits>
#include <utility>

#include <sirius/buffer.hpp>
#include <sirius/device.hpp>
#include <sirius/mrc_io.hpp>
#include <sirius/tiff_io.hpp>

#include "core/array_codec.hpp"
#include "core/array_source.hpp"
#include "core/build_info.hpp"
#include "core/host.hpp"
#include "core/manifest.hpp"

namespace sirius::app {

    namespace fs = std::filesystem;
    using json = nlohmann::json;

    std::size_t HostArray::elements() const noexcept {
        std::size_t n = 1;
        for (Index s : shape) n *= static_cast<std::size_t>(std::max<Index>(s, 0));
        return n;
    }

    namespace {

        // --- element types ------------------------------------------------------------------

        // Calls f(T{}) with the C++ type of `dtype` ("bool" is one byte, as uint8).
        template <typename F> decltype(auto) withDtype(const std::string& dtype, F&& f) {
            if (dtype == "uint8" || dtype == "bool") return f(std::uint8_t{});
            if (dtype == "int8") return f(std::int8_t{});
            if (dtype == "uint16") return f(std::uint16_t{});
            if (dtype == "int16") return f(std::int16_t{});
            if (dtype == "uint32") return f(std::uint32_t{});
            if (dtype == "int32") return f(std::int32_t{});
            if (dtype == "uint64") return f(std::uint64_t{});
            if (dtype == "int64") return f(std::int64_t{});
            if (dtype == "float64") return f(double{});
            if (dtype == "float32") return f(float{});
            throw DatasetError("unsupported dtype '" + dtype + "'");
        }

        // The same for the pixel types a TIFF decodes to.
        template <typename F> decltype(auto) withPixelDtype(const std::string& dtype, F&& f) {
            if (dtype == "uint8") return f(std::uint8_t{});
            if (dtype == "int8") return f(std::int8_t{});
            if (dtype == "uint16") return f(std::uint16_t{});
            if (dtype == "int16") return f(std::int16_t{});
            if (dtype == "uint32") return f(std::uint32_t{});
            if (dtype == "int32") return f(std::int32_t{});
            if (dtype == "float64") return f(double{});
            if (dtype == "float32") return f(float{});
            throw DatasetError("a TIFF of pixel type '" + dtype + "'");
        }

        const char* dtypeOf(PixelType t) {
            switch (t) {
                case PixelType::UInt8: return "uint8";
                case PixelType::Int8: return "int8";
                case PixelType::UInt16: return "uint16";
                case PixelType::Int16: return "int16";
                case PixelType::UInt32: return "uint32";
                case PixelType::Int32: return "int32";
                case PixelType::Float32: return "float32";
                case PixelType::Float64: return "float64";
            }
            return "float32";
        }

        template <typename T> T* as(HostArray& a) { return reinterpret_cast<T*>(a.bytes.data()); }
        template <typename T> const T* as(const HostArray& a) { return reinterpret_cast<const T*>(a.bytes.data()); }

        HostArray make(const std::string& dtype, std::vector<Index> shape) {
            HostArray a;
            a.dtype = dtype;
            a.shape = std::move(shape);
            a.bytes.resize(a.elements() * codec::dtypeSize(dtype));
            return a;
        }

        // The array as float32 (the wire takes no "bool"; a sample for the window statistics).
        std::vector<float> toFloat(const HostArray& a, std::size_t first = 0, std::size_t count = std::numeric_limits<std::size_t>::max(),
                                   std::size_t stride = 1) {
            std::vector<float> out;
            const std::size_t n = a.elements();
            withDtype(a.dtype, [&](auto tag) {
                using T = decltype(tag);
                const T* p = as<T>(a);
                for (std::size_t i = first, k = 0; i < n && k < count; i += stride, ++k) out.push_back(static_cast<float>(p[i]));
            });
            return out;
        }

        // Rows [r0, r1) x columns [c0, c1) of a 2-D array.
        HostArray crop2(const HostArray& a, Index r0, Index r1, Index c0, Index c1) {
            const std::size_t item = codec::dtypeSize(a.dtype);
            HostArray out = make(a.dtype, {r1 - r0, c1 - c0});
            const Index cols = a.shape[1];
            for (Index r = r0; r < r1; ++r)
                std::memcpy(out.bytes.data() + static_cast<std::size_t>((r - r0) * (c1 - c0)) * item,
                            a.bytes.data() + static_cast<std::size_t>(r * cols + c0) * item, static_cast<std::size_t>(c1 - c0) * item);
            return out;
        }

        // Plane k of a (n, y, x) array as (y, x).
        HostArray planeOf(const HostArray& a, Index k) {
            const std::size_t item = codec::dtypeSize(a.dtype);
            const std::size_t plane = static_cast<std::size_t>(a.shape[1] * a.shape[2]) * item;
            HostArray out;
            out.dtype = a.dtype;
            out.shape = {a.shape[1], a.shape[2]};
            out.bytes.assign(a.bytes.begin() + static_cast<std::ptrdiff_t>(plane * static_cast<std::size_t>(k)),
                             a.bytes.begin() + static_cast<std::ptrdiff_t>(plane * static_cast<std::size_t>(k + 1)));
            return out;
        }

        int ceilDiv(Index a, Index b) { return static_cast<int>((a + b - 1) / b); }

    } // namespace

    // --- reduce_blocks --------------------------------------------------------------------------

    HostArray reduceBlocks(const HostArray& a, const std::vector<int>& factorsIn) {
        if (factorsIn.size() != a.shape.size()) throw std::invalid_argument("reduceBlocks: one factor per axis");
        std::vector<Index> f(factorsIn.size());
        bool identity = true;
        for (std::size_t k = 0; k < f.size(); ++k) {
            f[k] = std::max(factorsIn[k], 1);
            identity = identity && f[k] == 1;
        }
        if (identity) return a;
        const std::size_t rank = a.shape.size();
        if (rank < 1 || rank > 3) throw std::invalid_argument("reduceBlocks: 1 to 3 axes");
        // as 3-D (padded with leading axes of 1)
        std::array<Index, 3> n{1, 1, 1}, ff{1, 1, 1};
        for (std::size_t k = 0; k < rank; ++k) {
            n[3 - rank + k] = a.shape[k];
            ff[3 - rank + k] = f[k];
        }
        std::array<Index, 3> o{};
        for (int k = 0; k < 3; ++k) o[static_cast<std::size_t>(k)] = (n[static_cast<std::size_t>(k)] + ff[static_cast<std::size_t>(k)] - 1) / ff[static_cast<std::size_t>(k)];
        std::vector<double> sum(static_cast<std::size_t>(o[0] * o[1] * o[2]), 0.0);
        std::vector<Index> outShape(rank);
        for (std::size_t k = 0; k < rank; ++k) outShape[k] = o[3 - rank + k];
        HostArray out = make(a.dtype, outShape);
        withDtype(a.dtype, [&](auto tag) {
            using T = decltype(tag);
            const T* p = as<T>(a);
            // exact for every integer type (float64 holds 2^53); float data in float64 as numpy sums it
            for (Index z = 0; z < n[0]; ++z)
                for (Index y = 0; y < n[1]; ++y) {
                    const std::size_t orow = static_cast<std::size_t>(((z / ff[0]) * o[1] + y / ff[1]) * o[2]);
                    const T* row = p + (z * n[1] + y) * n[2];
                    for (Index x = 0; x < n[2]; ++x) sum[orow + static_cast<std::size_t>(x / ff[2])] += static_cast<double>(row[x]);
                }
            T* q = as<T>(out);
            for (Index z = 0; z < o[0]; ++z)
                for (Index y = 0; y < o[1]; ++y)
                    for (Index x = 0; x < o[2]; ++x) {
                        const Index cz = std::min(ff[0], n[0] - z * ff[0]), cy = std::min(ff[1], n[1] - y * ff[1]), cx = std::min(ff[2], n[2] - x * ff[2]);
                        const std::size_t i = static_cast<std::size_t>((z * o[1] + y) * o[2] + x);
                        if constexpr (std::is_floating_point_v<T>) {
                            q[i] = static_cast<T>(sum[i] / static_cast<double>(cz * cy * cx));
                        } else {
                            // numpy: the float32 sum over the float32 count, rounded half to even, clipped
                            const float mean = static_cast<float>(sum[i]) / static_cast<float>(cz * cy * cx);
                            double r = std::nearbyint(static_cast<double>(mean));
                            r = std::clamp(r, static_cast<double>(std::numeric_limits<T>::lowest()), static_cast<double>(std::numeric_limits<T>::max()));
                            q[i] = static_cast<T>(r);
                        }
                    }
        });
        return out;
    }

    // --- the datasets ---------------------------------------------------------------------------

    namespace {

        // A step output's handle (core/remote_source.hpp, makeOutputHandle).
        constexpr const char* kOutputPrefix = "sirius-out:";

        std::string lowerTrim(const std::string& s) {
            std::string out;
            for (char ch : s) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            const auto space = [](char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'; };
            while (!out.empty() && space(out.front())) out.erase(out.begin());
            while (!out.empty() && space(out.back())) out.pop_back();
            return out;
        }

        bool endsWith(const std::string& s, const std::string& tail) {
            return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
        }

        std::string hostName() {
            const std::string h = host::hostName();
            return h.empty() ? std::string("this machine") : h;
        }

        // The CUDA device a decode named `device` runs on with nvTIFF, or none
        // (a CPU device, a build without nvTIFF, no such GPU): datasets.py _gpu.
        std::optional<Device> gpuFor(const std::string& device) {
            const std::string text = lowerTrim(device.empty() ? std::string("auto") : device);
            if (text != "auto" && text.rfind("cuda", 0) != 0) return std::nullopt;
            if (!builtWithNvTiff() || !cudaAvailable()) return std::nullopt;
            int index = 0;
            if (text.rfind("cuda:", 0) == 0) {
                const std::string digits = text.substr(5);
                if (digits.empty() || digits.size() > 4 || !std::all_of(digits.begin(), digits.end(), [](char ch) { return ch >= '0' && ch <= '9'; }))
                    return std::nullopt;
                index = std::atoi(digits.c_str());
            }
            if (index < 0 || index >= cudaDeviceCount()) return std::nullopt;
            return Device::cuda(index);
        }

        // (page_order, c, t, z) of a request's options, as datasets.py keys
        // them, and the tile of a folder dataset.
        struct OptionsKey {
            std::optional<std::string> order;
            Index c = 0, t = 0, z = 0, tile = 0;
            std::string text() const {
                return (order ? "o:" + *order : std::string("-")) + "|" + std::to_string(c) + "|" + std::to_string(t) + "|" + std::to_string(z) +
                       (tile > 0 ? "|tile " + std::to_string(tile) : std::string());
            }
        };

        Index intOf(const json& v, Index def) {
            if (v.is_number_integer()) return v.get<Index>();
            if (v.is_number_float()) return static_cast<Index>(v.get<double>());
            if (v.is_boolean()) return v.get<bool>() ? 1 : 0;
            if (v.is_string()) {
                const std::string s = v.get<std::string>();
                char* end = nullptr;
                const long long x = std::strtoll(s.c_str(), &end, 10);
                if (end && *end == '\0' && !s.empty()) return static_cast<Index>(x);
                throw DatasetError("'" + s + "' is not a number");
            }
            if (v.is_null()) return def;
            throw DatasetError(v.dump() + " is not a number");
        }

        // params.get(key, def) as int(); with `orDefault`, a falsy value (0, null) is the default too
        Index intParam(const json& p, const char* key, Index def, bool orDefault = false) {
            const auto it = p.find(key);
            if (it == p.end()) return def;
            const Index v = intOf(*it, def);
            return orDefault && v == 0 ? def : v;
        }

        OptionsKey optionsKey(const json& o) {
            OptionsKey k;
            if (!o.is_object()) return k;
            if (auto it = o.find("page_order"); it != o.end() && !it->is_null()) k.order = it->is_string() ? it->get<std::string>() : it->dump();
            k.c = o.contains("c") ? intOf(o["c"], 0) : 0;
            k.t = o.contains("t") ? intOf(o["t"], 0) : 0;
            k.z = o.contains("z") ? intOf(o["z"], 0) : 0;
            k.tile = o.contains("tile") ? std::max<Index>(intOf(o["tile"], 0), 0) : 0;
            return k;
        }

        // --- .npy -------------------------------------------------------------------------------

        struct NpyHeader {
            std::string dtype;        // after conversion: what a read returns ("float16" reads as float32)
            std::string fileDtype;    // numpy's name ("float16", "bool", ...)
            std::size_t item = 1;     // bytes per element in the file
            bool bigEndian = false;
            bool float16 = false;
            std::vector<Index> shape;
            std::uint64_t dataOffset = 0;
        };

        NpyHeader readNpyHeader(const std::string& path) {
            const std::string name = fs::u8path(path).filename().u8string();
            std::ifstream in(fs::u8path(path), std::ios::binary);
            if (!in) throw DatasetError(path + ": cannot be read");
            char magic[8];
            if (!in.read(magic, 8) || std::memcmp(magic, "\x93NUMPY", 6) != 0) throw DatasetError(name + ": not a .npy file");
            const int major = static_cast<unsigned char>(magic[6]);
            std::uint32_t hlen = 0;
            std::uint64_t offset = 10;
            if (major == 1) {
                unsigned char b[2];
                if (!in.read(reinterpret_cast<char*>(b), 2)) throw DatasetError(name + ": a truncated .npy header");
                hlen = b[0] | (b[1] << 8);
            } else if (major == 2 || major == 3) {
                unsigned char b[4];
                if (!in.read(reinterpret_cast<char*>(b), 4)) throw DatasetError(name + ": a truncated .npy header");
                hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
                offset = 12;
            } else {
                throw DatasetError(name + ": .npy format version " + std::to_string(major) + " is not read");
            }
            if (hlen > (1u << 20)) throw DatasetError(name + ": a .npy header of " + std::to_string(hlen) + " bytes");
            std::string h(hlen, '\0');
            if (!in.read(h.data(), hlen)) throw DatasetError(name + ": a truncated .npy header");
            NpyHeader r;
            r.dataOffset = offset + hlen;
            const auto valueOf = [&](const std::string& key) -> std::string {
                const std::size_t at = h.find("'" + key + "'");
                if (at == std::string::npos) throw DatasetError(name + ": the .npy header has no '" + key + "'");
                std::size_t p = h.find(':', at);
                if (p == std::string::npos) throw DatasetError(name + ": a malformed .npy header");
                ++p;
                while (p < h.size() && h[p] == ' ') ++p;
                std::size_t e = p;
                if (p < h.size() && (h[p] == '\'' || h[p] == '"')) {
                    e = h.find(h[p], p + 1);
                    return h.substr(p + 1, e - p - 1);
                }
                if (p < h.size() && h[p] == '(') return h.substr(p + 1, h.find(')', p) - p - 1);
                while (e < h.size() && h[e] != ',' && h[e] != '}') ++e;
                return h.substr(p, e - p);
            };
            const std::string descr = valueOf("descr");
            const std::string fortran = valueOf("fortran_order");
            if (fortran.find("True") != std::string::npos)
                throw DatasetError(name + ": a Fortran-ordered .npy array; save it in C order (numpy.ascontiguousarray)");
            std::string shapeText = valueOf("shape");
            for (std::size_t i = 0, s = 0; i <= shapeText.size(); ++i)
                if (i == shapeText.size() || shapeText[i] == ',') {
                    std::string part = shapeText.substr(s, i - s);
                    part.erase(std::remove(part.begin(), part.end(), ' '), part.end());
                    if (!part.empty()) {
                        char* end = nullptr;
                        const long long v = std::strtoll(part.c_str(), &end, 10);
                        if (!end || *end != '\0' || v < 0) throw DatasetError(name + ": a malformed .npy shape (" + shapeText + ")");
                        r.shape.push_back(static_cast<Index>(v));
                    }
                    s = i + 1;
                }
            if (descr.size() < 3) throw DatasetError(name + ": the .npy dtype '" + descr + "' is not read");
            const char order = descr[0], kind = descr[1];
            const int size = std::atoi(descr.c_str() + 2);
            r.bigEndian = order == '>';
            r.item = static_cast<std::size_t>(size);
            if (kind == 'b' && size == 1) r.fileDtype = r.dtype = "bool";
            else if (kind == 'u' && (size == 1 || size == 2 || size == 4 || size == 8)) r.fileDtype = r.dtype = "uint" + std::to_string(size * 8);
            else if (kind == 'i' && (size == 1 || size == 2 || size == 4 || size == 8)) r.fileDtype = r.dtype = "int" + std::to_string(size * 8);
            else if (kind == 'f' && (size == 4 || size == 8)) r.fileDtype = r.dtype = "float" + std::to_string(size * 8);
            else if (kind == 'f' && size == 2) {
                r.fileDtype = "float16";
                r.dtype = "float32";
                r.float16 = true;
            } else {
                throw DatasetError(name + ": the .npy dtype '" + descr + "' is not read");
            }
            return r;
        }

        float halfToFloat(std::uint16_t h) {
            const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
            std::uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu, bits;
            if (exp == 0) {
                if (mant == 0) bits = sign;
                else {
                    exp = 127 - 15 + 1;
                    while (!(mant & 0x400u)) {
                        mant <<= 1;
                        --exp;
                    }
                    mant &= 0x3ffu;
                    bits = sign | (exp << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                bits = sign | 0x7f800000u | (mant << 13);
            } else {
                bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
            }
            float f;
            std::memcpy(&f, &bits, sizeof f);
            return f;
        }

        // --- the cache of volumes (datasets.py _ByteLru) ----------------------------------------------

        class VolumeCache {
        public:
            explicit VolumeCache(std::size_t budget) : budget_(budget) {}
            std::shared_ptr<const HostArray> get(const std::string& key) {
                const std::lock_guard<std::mutex> g(m_);
                auto it = index_.find(key);
                if (it == index_.end()) return nullptr;
                order_.splice(order_.end(), order_, it->second);
                return it->second->second;
            }
            void put(const std::string& key, std::shared_ptr<const HostArray> value) {
                if (value->bytes.size() > budget_) return;
                const std::lock_guard<std::mutex> g(m_);
                if (auto it = index_.find(key); it != index_.end()) {
                    used_ -= it->second->second->bytes.size();
                    order_.erase(it->second);
                    index_.erase(it);
                }
                used_ += value->bytes.size();
                order_.emplace_back(key, std::move(value));
                index_[key] = std::prev(order_.end());
                while (used_ > budget_ && !order_.empty()) {
                    used_ -= order_.front().second->bytes.size();
                    index_.erase(order_.front().first);
                    order_.pop_front();
                }
            }
            void clear() {
                const std::lock_guard<std::mutex> g(m_);
                order_.clear();
                index_.clear();
                used_ = 0;
            }
            std::size_t used() const {
                const std::lock_guard<std::mutex> g(m_);
                return used_;
            }

        private:
            using Order = std::list<std::pair<std::string, std::shared_ptr<const HostArray>>>;
            std::size_t budget_;
            std::size_t used_ = 0;
            mutable std::mutex m_;
            Order order_;
            std::map<std::string, Order::iterator> index_;
        };

        std::size_t cacheBudget(long long given) {
            if (given >= 0) return static_cast<std::size_t>(given);
            const std::string env = host::environment("SIRIUS_WORKER_VIEW_CACHE_MB");
            if (!env.empty()) {
                char* end = nullptr;
                const long long mb = std::strtoll(env.c_str(), &end, 10);
                if (end && *end == '\0') return static_cast<std::size_t>(std::max<long long>(mb, 0)) << 20;
            }
            return std::size_t{4096} << 20;
        }

        // (x0, y0, x1, y1) of `region` (x, y, w, h) clipped to a cols x rows view; the whole view without one.
        std::array<Index, 4> boxOf(const json& region, Index cols, Index rows) {
            if (!region.is_array() || region.size() != 4) return {0, 0, cols, rows};
            Index x0 = intOf(region[0], 0), y0 = intOf(region[1], 0);
            const Index w = intOf(region[2], 0), h = intOf(region[3], 0);
            x0 = std::max<Index>(x0, 0);
            y0 = std::max<Index>(y0, 0);
            const Index x1 = std::min(x0 + std::max<Index>(w, 0), cols), y1 = std::min(y0 + std::max<Index>(h, 0), rows);
            if (x1 <= x0 || y1 <= y0) {
                std::string list = "[";
                for (std::size_t i = 0; i < 4; ++i) list += (i ? ", " : "") + std::to_string(intOf(region[i], 0));
                throw DatasetError("region " + list + "] is outside the " + std::to_string(cols) + " x " + std::to_string(rows) + " view");
            }
            return {x0, y0, x1, y1};
        }

    } // namespace

    // --- one dataset ------------------------------------------------------------------------------

    namespace {

        class Dataset {
        public:
            // A step's output the engine holds (its handle as the path): float32, read through its source.
            Dataset(std::string handle, DatasetService::ResolvedOutput output, VolumeCache& cache)
                : path_(handle), key_(std::move(handle)), cache_(cache), source_(std::move(output.source)), sourceMeta_(std::move(output.meta)) {
                if (!source_) throw DatasetError(path_ + ": no data");
                const Dims5& d = sourceMeta_.dims;
                c_ = d.c;
                t_ = d.t;
                z_ = d.z;
                y_ = d.y;
                x_ = d.x;
                fileDtype_ = "float32";
                format_ = "sirius-output";
                bytesOnDisk_ = static_cast<std::uint64_t>(std::max<Index>(d.numel(), 0)) * sizeof(float);
                voxel_ = sourceMeta_.voxelUm;
                frameInterval_ = std::max(sourceMeta_.frameIntervalS, 0.0);
                rgb_ = sourceMeta_.rgb;
                dimsFromMetadata_ = true;
                for (const ChannelInfo& ch : sourceMeta_.channels) {
                    json e = {{"name", ch.label}};
                    if (ch.wavelengthNm > 0.0) e["wavelength_nm"] = ch.wavelengthNm;
                    channels_.push_back(std::move(e));
                }
            }

            Dataset(std::string path, const json& options, VolumeCache& cache, std::string key)
                : path_(std::move(path)), key_(std::move(key)), cache_(cache) {
                std::error_code ec;
                const fs::path p = fs::u8path(path_);
                if (!fs::exists(p, ec)) throw DatasetError(path_ + ": no such file on " + hostName());
                // A folder dataset (its manifest, a manifest .toml, or a folder of
                // TIFF stacks), a zarr / N5 store: opened by the core's own code
                // (openDataset), as the application opens one on its computer,
                // so both give the same dims and values.
                if (fs::is_directory(p, ec) || isDatasetManifestFile(path_)) {
                    openFolder(options);
                    return;
                }
                bytesOnDisk_ = static_cast<std::uint64_t>(fs::file_size(p, ec));
                std::string lower;
                for (char ch : path_) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
                if (endsWith(lower, ".npy")) openNpy();
                else if (endsWith(lower, ".tif") || endsWith(lower, ".tiff") || endsWith(lower, ".btf") || endsWith(lower, ".tf8")) openTiff(options);
                else if (endsWith(lower, ".dv") || endsWith(lower, ".mrc")) openMrc(options);
                else throw DatasetError(p.filename().u8string() + ": only TIFF / OME-TIFF, DeltaVision / MRC (.dv, .mrc) and .npy are read on the cluster");
            }

            json meta() const {
                if (source_) {
                    json m = {{"name", sourceMeta_.name},
                              {"path", path_},
                              {"format", format_},
                              {"dims", {c_, t_, z_, y_, x_}},
                              {"dtype", fileDtype_},
                              {"bytes", bytesOnDisk_},
                              {"voxel_um", {voxel_[0], voxel_[1], voxel_[2]}},
                              {"frame_interval_s", frameInterval_},
                              {"channels", channels_},
                              {"rgb", rgb_},
                              {"dims_from_metadata", dimsFromMetadata_}};
                    // a folder dataset's tiles, and the one served
                    if (sourceMeta_.hasTiles()) {
                        json tiles = json::array();
                        for (const TileInfo& t : sourceMeta_.tiles)
                            tiles.push_back({{"name", t.name},
                                             {"position_um", {t.positionUm[0], t.positionUm[1], t.positionUm[2]}},
                                             {"grid_index", {t.gridIndex[0], t.gridIndex[1], t.gridIndex[2]}}});
                        m["tiles"] = std::move(tiles);
                        m["tile"] = sourceMeta_.tileIndex;
                    }
                    if (!sourceMeta_.acquisition.empty()) m["acquisition"] = sourceMeta_.acquisition;
                    return m;
                }
                std::string name = fs::u8path(path_).filename().u8string();
                std::string lname;
                for (char ch : name) lname.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
                for (const char* ext : {".ome.tiff", ".ome.tif", ".tiff", ".tif", ".npy", ".dv", ".mrc"})
                    if (endsWith(lname, ext)) {
                        name = name.substr(0, name.size() - std::strlen(ext));
                        break;
                    }
                return {{"name", name},
                        {"path", path_},
                        {"format", format_},
                        {"dims", {c_, t_, z_, y_, x_}},
                        {"dtype", fileDtype_},
                        {"bytes", bytesOnDisk_},
                        {"voxel_um", {voxel_[0], voxel_[1], voxel_[2]}},
                        {"frame_interval_s", frameInterval_},
                        {"channels", channels_},
                        {"rgb", rgb_},
                        {"dims_from_metadata", dimsFromMetadata_}};
            }

            Index c() const noexcept { return c_; }
            Index t() const noexcept { return t_; }
            Index z() const noexcept { return z_; }

            // --- reading, in the file's dtype ---------------------------------------------------

            HostArray plane(Index c, Index t, Index z, const std::string& device) {
                check(c, t, z);
                if (source_) {
                    HostArray a = make("float32", {y_, x_});
                    source_->readPlane(c, t, z, as<float>(a));
                    return a;
                }
                if (npy_) return readNpy(c, t, z, 1, true);
                if (mrc_) return readMrc(c, t, z, 1, true);
                if (auto cached = cache_.get(volumeKey(c, t))) return planeOf(*cached, z);
                return planeOf(pages(pageOf(c, t, z), 1, device, c), 0);
            }

            std::shared_ptr<const HostArray> volume(Index c, Index t, const std::string& device) {
                check(c, t, std::nullopt);
                const std::string key = volumeKey(c, t);
                if (auto cached = cache_.get(key)) return cached;
                // A step's output: kept like a file's volume (within the same
                // budget), so panning a re-slice does not copy it each time.
                if (source_) {
                    auto vol = std::make_shared<HostArray>(make("float32", {z_, y_, x_}));
                    source_->readVolume(c, t, as<float>(*vol));
                    cache_.put(key, vol);
                    return vol;
                }
                auto vol = std::make_shared<HostArray>(npy_ ? readNpy(c, t, 0, z_, false) : mrc_ ? readMrc(c, t, 0, z_, false) : tiffVolume(c, t, device));
                cache_.put(key, vol);
                return vol;
            }

            HostArray view(const std::string& kindIn, Index c, Index t, Index index, Index factorIn, const json& region, Index maxSide,
                           const std::string& device) {
                const Index factor = std::max<Index>(factorIn, 1);
                const std::string kind = lowerTrim(kindIn);
                const int f = static_cast<int>(std::min<Index>(factor, std::numeric_limits<int>::max()));
                if (kind == "volume") {
                    auto vol = volume(c, t, device);
                    std::vector<int> fs3;
                    for (Index n : vol->shape) fs3.push_back(static_cast<int>(std::max<Index>(1, (n + std::max<Index>(maxSide, 1) - 1) / std::max<Index>(maxSide, 1))));
                    return reduceBlocks(*vol, fs3);
                }
                HostArray full;
                if (kind == "xy") {
                    check(c, t, index);
                    auto cached = cache_.get(volumeKey(c, t));
                    if (!cached && tiff_) return tiffXY(c, t, index, factor, boxOf(region, x_, y_), device);
                    full = cached ? planeOf(*cached, index) : plane(c, t, index, device);
                } else if (kind == "xz") {
                    check(c, t, std::nullopt);
                    auto vol = volume(c, t, device);
                    const Index y = std::clamp<Index>(index, 0, y_ - 1);
                    full = make(vol->dtype, {z_, x_});
                    const std::size_t item = codec::dtypeSize(vol->dtype), row = static_cast<std::size_t>(x_) * item;
                    for (Index z = 0; z < z_; ++z)
                        std::memcpy(full.bytes.data() + static_cast<std::size_t>(z) * row, vol->bytes.data() + static_cast<std::size_t>((z * y_ + y) * x_) * item, row);
                } else if (kind == "yz") {
                    check(c, t, std::nullopt);
                    auto vol = volume(c, t, device);
                    const Index x = std::clamp<Index>(index, 0, x_ - 1);
                    full = make(vol->dtype, {y_, z_});
                    const std::size_t item = codec::dtypeSize(vol->dtype);
                    for (Index y = 0; y < y_; ++y)
                        for (Index z = 0; z < z_; ++z)
                            std::memcpy(full.bytes.data() + static_cast<std::size_t>(y * z_ + z) * item, vol->bytes.data() + static_cast<std::size_t>((z * y_ + y) * x_ + x) * item, item);
                } else if (kind == "mip") {
                    check(c, t, std::nullopt);
                    const std::string key = volumeKey(c, t) + "|mip";
                    auto cached = cache_.get(key);
                    if (!cached) {
                        auto vol = volume(c, t, device);
                        auto mip = std::make_shared<HostArray>(make(vol->dtype, {y_, x_}));
                        withDtype(vol->dtype, [&](auto tag) {
                            using T = decltype(tag);
                            const T* p = as<T>(*vol);
                            T* q = as<T>(*mip);
                            const std::size_t plane = static_cast<std::size_t>(y_ * x_);
                            for (std::size_t i = 0; i < plane; ++i) {
                                T m = p[i];
                                for (Index z = 1; z < z_; ++z) {
                                    const T v = p[static_cast<std::size_t>(z) * plane + i];
                                    if constexpr (std::is_floating_point_v<T>) {
                                        if (std::isnan(m)) break;   // numpy's max propagates NaN
                                        if (std::isnan(v) || v > m) m = v;
                                    } else if (v > m) {
                                        m = v;
                                    }
                                }
                                q[i] = m;
                            }
                        });
                        cache_.put(key, mip);
                        cached = mip;
                    }
                    full = *cached;
                } else {
                    throw DatasetError("unknown view '" + kindIn + "': xy, xz, yz, mip or volume");
                }
                const Index rows = full.shape[0], cols = full.shape[1];
                const auto [x0, y0, x1, y1] = boxOf(region, cols, rows);
                HostArray part = (x0 == 0 && y0 == 0 && x1 == cols && y1 == rows) ? std::move(full) : crop2(full, y0, y1, x0, x1);
                return reduceBlocks(part, {f, f});
            }

            json stats(Index c, Index t, const std::string& device) {
                check(c, t, std::nullopt);
                const Index n = std::min<Index>(z_, 5);
                std::vector<Index> zs;
                if (n == 1) zs = {0};
                else
                    for (Index k = 0; k < n; ++k) zs.push_back(k * (z_ - 1) / (n - 1));
                std::vector<float> s;
                for (Index z : zs) {
                    const HostArray p = plane(c, t, z, device);
                    const std::size_t size = p.elements();
                    const std::size_t stride = std::max<std::size_t>(1, size / (std::size_t{1} << 16));
                    const std::vector<float> part = toFloat(p, 0, std::numeric_limits<std::size_t>::max(), stride);
                    s.insert(s.end(), part.begin(), part.end());
                }
                s.erase(std::remove_if(s.begin(), s.end(), [](float v) { return !std::isfinite(v); }), s.end());
                if (s.empty()) return {{"lo", 0.0}, {"hi", 1.0}, {"min", 0.0}, {"max", 1.0}};
                std::sort(s.begin(), s.end());
                // numpy's linear percentile: the virtual index (n - 1) q, and the
                // difference of the neighbours taken in float32
                const auto percentile = [&](double q) {
                    const double vi = (static_cast<double>(s.size()) - 1.0) * (q / 100.0);
                    const double lo = std::floor(vi);
                    const double g = vi - lo;
                    const std::size_t i = static_cast<std::size_t>(lo), j = std::min(i + 1, s.size() - 1);
                    const float a = s[i], b = s[j];
                    const float diff = b - a;
                    return g >= 0.5 ? static_cast<double>(b) - static_cast<double>(diff) * (1.0 - g) : static_cast<double>(a) + static_cast<double>(diff) * g;
                };
                double lo = percentile(0.1), hi = percentile(99.9);
                const double mn = s.front(), mx = s.back();
                if (hi <= lo) {
                    lo = mn;
                    hi = mx > mn ? mx : mn + 1.0;
                }
                return {{"lo", lo}, {"hi", hi}, {"min", mn}, {"max", mx}};
            }

        private:
            void check(Index c, Index t, std::optional<Index> z) const {
                if (!(c >= 0 && c < c_ && t >= 0 && t < t_) || (z && !(*z >= 0 && *z < z_))) {
                    const std::string where = "c " + std::to_string(c) + ", t " + std::to_string(t) + (z ? ", z " + std::to_string(*z) : std::string());
                    throw DatasetError(where + " is outside the dataset (" + std::to_string(c_) + " channels, " + std::to_string(t_) + " time points, " +
                                       std::to_string(z_) + " planes)");
                }
            }
            std::string volumeKey(Index c, Index t) const { return key_ + "|" + std::to_string(c) + "|" + std::to_string(t); }

            // --- opening ------------------------------------------------------------------------

            // through openDataset, lazily: planes and volumes are read from the
            // folder's files on demand, as float32
            void openFolder(const json& options) {
                OpenOptions o;
                o.readAll = false;
                const OptionsKey k = optionsKey(options);
                o.tile = k.tile;
                if (k.order) {
                    PageOrder po;
                    po.order = *k.order;
                    po.c = k.c;
                    po.t = k.t;
                    po.z = k.z;
                    o.pageOrder = po;
                }
                OpenResult opened;
                try {
                    opened = openDataset(path_, o);
                } catch (const std::exception& e) {
                    throw DatasetError(fs::u8path(path_).filename().u8string() + ": " + e.what());
                }
                source_ = opened.source;
                sourceMeta_ = opened.meta;
                const Dims5& d = sourceMeta_.dims;
                c_ = d.c;
                t_ = d.t;
                z_ = d.z;
                y_ = d.y;
                x_ = d.x;
                fileDtype_ = dtypeOf(sourceMeta_.sourceType);
                format_ = sourceMeta_.format.empty() ? std::string("folder") : sourceMeta_.format;
                bytesOnDisk_ = sourceMeta_.bytesOnDisk;
                voxel_ = sourceMeta_.voxelUm;
                frameInterval_ = std::max(sourceMeta_.frameIntervalS, 0.0);
                rgb_ = sourceMeta_.rgb;
                dimsFromMetadata_ = opened.dimsFromMetadata;
                for (const ChannelInfo& ch : sourceMeta_.channels) {
                    json e = {{"name", ch.label}};
                    if (ch.wavelengthNm > 0.0) e["wavelength_nm"] = ch.wavelengthNm;
                    channels_.push_back(std::move(e));
                }
            }

            void openNpy() {
                const NpyHeader h = readNpyHeader(path_);
                const std::string name = fs::u8path(path_).filename().u8string();
                std::vector<Index> s = h.shape;
                if (s.size() == 2) s = {1, 1, 1, s[0], s[1]};
                else if (s.size() == 3) s = {1, 1, s[0], s[1], s[2]};
                else if (s.size() == 4) s = {s[0], 1, s[1], s[2], s[3]};
                else if (s.size() != 5)
                    throw DatasetError(name + ": a " + std::to_string(h.shape.size()) + "-D array; 2 to 5 dimensions (c, t, z, y, x) are read");
                std::uint64_t need = h.item;
                for (Index v : s) need *= static_cast<std::uint64_t>(v);
                if (bytesOnDisk_ < h.dataOffset + need) throw DatasetError(name + ": the .npy file is shorter than its header says");
                npy_ = h;
                c_ = s[0];
                t_ = s[1];
                z_ = s[2];
                y_ = s[3];
                x_ = s[4];
                fileDtype_ = h.fileDtype;
                format_ = "npy";
                dimsFromMetadata_ = h.shape.size() > 3;
            }

            void openTiff(const json& options) {
                const std::string name = fs::u8path(path_).filename().u8string();
                try {
                    tiff_ = std::make_unique<TiffFile>(path_);
                } catch (const std::exception& e) {
                    throw DatasetError(name + ": " + e.what());
                }
                const TiffInfo& info = tiff_->info();
                if (!info.uniformPages()) throw DatasetError(name + ": the pages differ in size or pixel type");
                if (!info.page(0).decodable()) throw DatasetError(name + ": " + info.page(0).unsupported);
                OpenOptions o;
                if (options.is_object() && options.contains("page_order") && !options["page_order"].is_null()) {
                    PageOrder po;
                    po.order = options["page_order"].is_string() ? options["page_order"].get<std::string>() : std::string("czt");
                    po.c = options.contains("c") ? intOf(options["c"], 0) : 0;
                    po.t = options.contains("t") ? intOf(options["t"], 0) : 0;
                    po.z = options.contains("z") ? intOf(options["z"], 0) : 0;
                    o.pageOrder = po;
                }
                TiffDatasetProbe p;
                try {
                    p = probeTiffDataset(path_, info, &o);
                } catch (const std::exception& e) {
                    throw DatasetError(name + ": " + e.what());
                }
                order_ = p.order;
                samples_ = std::max<Index>(p.samples, 1);
                c_ = p.meta.dims.c;
                t_ = p.meta.dims.t;
                z_ = p.meta.dims.z;
                y_ = p.meta.dims.y;
                x_ = p.meta.dims.x;
                fileDtype_ = dtypeOf(info.pixelType());
                rgb_ = p.meta.rgb;
                dimsFromMetadata_ = p.dimsFromMetadata;
                format_ = p.parsed.ome ? "ome-tiff" : (p.parsed.imagej ? "imagej-tiff" : "tiff");
                voxel_ = p.fileVoxelUm;
                frameInterval_ = std::max(p.parsed.frameIntervalS, 0.0);
                if (p.parsed.ome && samples_ == 1)
                    for (const ChannelInfo& ch : p.parsed.channels) {
                        json e = {{"name", ch.label}};
                        if (ch.wavelengthNm > 0.0) e["wavelength_nm"] = ch.wavelengthNm;
                        channels_.push_back(std::move(e));
                    }
                for (const TiffLevel& lv : info.levels) levels_.push_back({static_cast<Index>(lv.width), static_cast<Index>(lv.height), static_cast<Index>(lv.ifds.size())});
            }

            // An MRC / DeltaVision stack, shaped as the application's Load step
            // shapes it (probeMrcDataset): the header's wavelengths and time
            // points, or the page order the request gives.
            void openMrc(const json& options) {
                const std::string name = fs::u8path(path_).filename().u8string();
                try {
                    mrc_ = std::make_unique<MrcFile>(path_);
                } catch (const std::exception& e) {
                    throw DatasetError(name + ": " + e.what());
                }
                OpenOptions o;
                if (options.is_object() && options.contains("page_order") && !options["page_order"].is_null()) {
                    PageOrder po;
                    po.order = options["page_order"].is_string() ? options["page_order"].get<std::string>() : std::string("czt");
                    po.c = options.contains("c") ? intOf(options["c"], 0) : 0;
                    po.t = options.contains("t") ? intOf(options["t"], 0) : 0;
                    po.z = options.contains("z") ? intOf(options["z"], 0) : 0;
                    o.pageOrder = po;
                }
                MrcDatasetProbe p;
                try {
                    p = probeMrcDataset(path_, mrc_->info(), &o);
                } catch (const std::exception& e) {
                    throw DatasetError(name + ": " + e.what());
                }
                order_ = p.order;
                samples_ = 1;
                c_ = p.meta.dims.c;
                t_ = p.meta.dims.t;
                z_ = p.meta.dims.z;
                y_ = p.meta.dims.y;
                x_ = p.meta.dims.x;
                fileDtype_ = dtypeOf(mrc_->info().pixelType);
                rgb_ = false;
                dimsFromMetadata_ = p.dimsFromMetadata;
                format_ = p.meta.format;
                voxel_ = p.fileVoxelUm;
                frameInterval_ = 0.0;
                for (const ChannelInfo& ch : p.fileChannels) {
                    json e = {{"name", ch.label}};
                    if (ch.wavelengthNm > 0.0) e["wavelength_nm"] = ch.wavelengthNm;
                    channels_.push_back(std::move(e));
                }
            }

            // --- .npy reads ---------------------------------------------------------------------------

            // `count` planes from (c, t, z); as (y, x) when `asPlane`, else (count, y, x)
            HostArray readNpy(Index c, Index t, Index z, Index count, bool asPlane) const {
                const NpyHeader& h = *npy_;
                const std::size_t planeElems = static_cast<std::size_t>(y_ * x_);
                const std::uint64_t first = static_cast<std::uint64_t>(((c * t_ + t) * z_ + z)) * planeElems;
                const std::size_t n = planeElems * static_cast<std::size_t>(count);
                std::vector<std::byte> raw(n * h.item);
                std::ifstream in(fs::u8path(path_), std::ios::binary);
                in.seekg(static_cast<std::streamoff>(h.dataOffset + first * h.item));
                if (!in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size())))
                    throw DatasetError(fs::u8path(path_).filename().u8string() + ": a read past the end of the file");
                if (h.bigEndian && h.item > 1)
                    for (std::size_t i = 0; i < n; ++i) std::reverse(raw.begin() + static_cast<std::ptrdiff_t>(i * h.item), raw.begin() + static_cast<std::ptrdiff_t>((i + 1) * h.item));
                HostArray a;
                a.shape = asPlane ? std::vector<Index>{y_, x_} : std::vector<Index>{count, y_, x_};
                if (h.float16) {
                    a.dtype = "float32";
                    a.bytes.resize(n * sizeof(float));
                    float* q = as<float>(a);
                    for (std::size_t i = 0; i < n; ++i) {
                        std::uint16_t v;
                        std::memcpy(&v, raw.data() + i * 2, 2);
                        q[i] = halfToFloat(v);
                    }
                } else {
                    a.dtype = h.dtype;
                    a.bytes = std::move(raw);
                }
                return a;
            }

            // --- MRC / DeltaVision reads --------------------------------------------------------------

            // `count` planes from (c, t, z) in the file's pixel type; as (y, x) when `asPlane`, else
            // (count, y, x). One read when the planes are consecutive sections, else one per plane.
            HostArray readMrc(Index c, Index t, Index z, Index count, bool asPlane) const {
                HostArray a = make(fileDtype_, asPlane ? std::vector<Index>{y_, x_} : std::vector<Index>{count, y_, x_});
                const Index first = order_.planeOf(c, t, z);
                const Index stride = z_ > 1 ? order_.planeOf(c, t, 1) - order_.planeOf(c, t, 0) : 1;
                withPixelDtype(fileDtype_, [&](auto tag) {
                    using T = decltype(tag);
                    T* dst = as<T>(a);
                    const std::size_t plane = static_cast<std::size_t>(y_ * x_);
                    if (stride == 1 || count == 1) {
                        mrc_->readSections<T>(static_cast<std::size_t>(first), static_cast<std::size_t>(count), dst);
                        return;
                    }
                    for (Index k = 0; k < count; ++k)
                        mrc_->readSections<T>(static_cast<std::size_t>(order_.planeOf(c, t, z + k)), 1, dst + static_cast<std::size_t>(k) * plane);
                });
                return a;
            }

            // --- TIFF reads -----------------------------------------------------------------------------

            Index pageOf(Index c, Index t, Index z) const { return order_.planeOf(c / samples_, t, z); }

            // the decode device: the GPU when nvTIFF decodes this file there (datasets.py _device)
            Device decodeDevice(const std::string& device) {
                const std::optional<Device> gpu = gpuFor(device);
                if (!gpu) return Device::cpu();
                const std::string key = lowerTrim(device.empty() ? std::string("auto") : device);
                const std::lock_guard<std::mutex> g(gpuMutex_);
                auto it = gpuOk_.find(key);
                if (it == gpuOk_.end()) {
                    bool ok = false;
                    try {
                        ok = tiff_->gpuDecodable(*gpu);
                    } catch (const std::exception&) {
                        ok = false;
                    }
                    it = gpuOk_.emplace(key, ok).first;
                }
                return it->second ? *gpu : Device::cpu();
            }

            HostArray decode(const std::vector<std::uint64_t>& ifds, Region region, const std::string& device, Index c) {
                TiffReadOptions opts;
                opts.device = decodeDevice(device);
                opts.allowCpuFallback = true;
                if (samples_ > 1) {
                    opts.firstSample = static_cast<std::uint16_t>(c % samples_);
                    opts.sampleCount = 1;
                }
                const std::string dtype = fileDtype_;
                return withPixelDtype(dtype, [&](auto tag) {
                    using T = decltype(tag);
                    const Shape shape = tiff_->readShape(ifds.size(), region.height, region.width, opts);
                    Buffer<T> out(shape, opts.device);
                    tiff_->decode<T>(ifds, region, out.view(), opts);
                    if (opts.device.isCuda()) out = out.to(Device::cpu());
                    HostArray a;
                    a.dtype = dtype;
                    a.shape = {static_cast<Index>(ifds.size()), static_cast<Index>(region.height), static_cast<Index>(region.width)};
                    a.bytes.resize(a.elements() * sizeof(T));
                    if (!a.bytes.empty()) std::memcpy(a.bytes.data(), out.data(), a.bytes.size());
                    return a;
                });
            }

            // pages [first, first + count) of channel c's sample as (count, y, x)
            HostArray pages(Index first, Index count, const std::string& device, Index c) {
                const TiffInfo& info = tiff_->info();
                const std::vector<std::uint64_t> ifds(info.pages.begin() + static_cast<std::ptrdiff_t>(first),
                                                      info.pages.begin() + static_cast<std::ptrdiff_t>(first + count));
                return decode(ifds, Region{0, 0, static_cast<std::uint32_t>(x_), static_cast<std::uint32_t>(y_)}, device, c);
            }

            // (h, w) at (x, y) of one page at pyramid `level` (channel c's sample)
            HostArray region(Index page, Index x, Index y, Index w, Index h, std::size_t level, const std::string& device, Index c) {
                const TiffLevel& lv = tiff_->info().levels.at(level);
                const Region r = Region{static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)}
                                     .resolve(lv.width, lv.height);
                HostArray a = decode({lv.ifds.at(static_cast<std::size_t>(page))}, r, device, c);
                a.shape = {static_cast<Index>(r.height), static_cast<Index>(r.width)};
                return a;
            }

            // (level, scale) of the most reduced pyramid level whose integer scale divides `factor`
            std::optional<std::pair<std::size_t, Index>> levelFor(Index factor) const {
                std::optional<std::pair<std::size_t, Index>> best;
                const Index pages = levels_.empty() ? 0 : levels_[0][2];
                for (std::size_t k = 1; k < levels_.size(); ++k) {
                    const Index w = levels_[k][0], h = levels_[k][1], n = levels_[k][2];
                    if (w <= 0 || h <= 0 || n != pages) continue;
                    const Index s = static_cast<Index>(std::nearbyint(static_cast<double>(x_) / static_cast<double>(w)));
                    if (s < 2 || factor % s != 0 || (w != x_ / s && w != ceilDiv(x_, s)) || (h != y_ / s && h != ceilDiv(y_, s))) continue;
                    if (!best || s > best->second) best = std::make_pair(k, s);
                }
                return best;
            }

            HostArray tiffXY(Index c, Index t, Index z, Index factor, const std::array<Index, 4>& box, const std::string& device) {
                const auto [x0, y0, x1, y1] = box;
                const Index page = pageOf(c, t, z);
                const int f = static_cast<int>(factor);
                if (factor > 1)
                    if (const auto lv = levelFor(factor); lv && x0 % lv->second == 0 && y0 % lv->second == 0) {
                        const auto [k, s] = *lv;
                        const Index w = levels_[k][0], h = levels_[k][1];
                        const Index lx0 = x0 / s, ly0 = y0 / s, lx1 = std::min<Index>(ceilDiv(x1, s), w), ly1 = std::min<Index>(ceilDiv(y1, s), h);
                        if (lx1 > lx0 && ly1 > ly0) {
                            const HostArray part = region(page, lx0, ly0, lx1 - lx0, ly1 - ly0, k, device, c);
                            HostArray out = reduceBlocks(part, {static_cast<int>(factor / s), static_cast<int>(factor / s)});
                            if (out.shape[0] == ceilDiv(y1 - y0, factor) && out.shape[1] == ceilDiv(x1 - x0, factor)) return out;
                        }
                    }
                HostArray full = (x0 == 0 && y0 == 0 && x1 == x_ && y1 == y_) ? planeOf(pages(page, 1, device, c), 0)
                                                                              : region(page, x0, y0, x1 - x0, y1 - y0, 0, device, c);
                return reduceBlocks(full, {f, f});
            }

            // the z planes of (c, t): one page range when z is the fastest axis; when other
            // axes' pages sit between them, one range of at most 4x (and 1 GiB) taken every
            // stride-th page, else page by page
            HostArray tiffVolume(Index c, Index t, const std::string& device) {
                const Index first = pageOf(c, t, 0);
                const Index stride = z_ > 1 ? pageOf(c, t, 1) - first : 1;
                if (stride == 1) return pages(first, z_, device, c);
                const Index span = (z_ - 1) * stride + 1;
                const std::size_t item = codec::dtypeSize(fileDtype_);
                const std::size_t planeBytes = static_cast<std::size_t>(y_ * x_) * item;
                HostArray vol = make(fileDtype_, {z_, y_, x_});
                if (stride > 0 && stride <= 4 && static_cast<std::uint64_t>(span) * planeBytes <= (std::uint64_t{1} << 30)) {
                    const HostArray all = pages(first, span, device, c);
                    for (Index z = 0; z < z_; ++z)
                        std::memcpy(vol.bytes.data() + static_cast<std::size_t>(z) * planeBytes, all.bytes.data() + static_cast<std::size_t>(z * stride) * planeBytes, planeBytes);
                    return vol;
                }
                for (Index z = 0; z < z_; ++z) {
                    const HostArray p = pages(first + z * stride, 1, device, c);
                    std::memcpy(vol.bytes.data() + static_cast<std::size_t>(z) * planeBytes, p.bytes.data(), planeBytes);
                }
                return vol;
            }

            std::string path_, key_;
            VolumeCache& cache_;
            std::uint64_t bytesOnDisk_ = 0;
            Index c_ = 1, t_ = 1, z_ = 1, y_ = 1, x_ = 1;
            std::string fileDtype_ = "float32", format_;
            std::array<double, 3> voxel_{0.0, 0.0, 0.0};
            double frameInterval_ = 0.0;
            json channels_ = json::array();
            bool rgb_ = false, dimsFromMetadata_ = false;
            std::optional<NpyHeader> npy_;
            std::unique_ptr<MrcFile> mrc_;
            std::unique_ptr<TiffFile> tiff_;
            PageOrder order_;
            Index samples_ = 1;
            std::vector<std::array<Index, 3>> levels_;   // (width, height, pages) per pyramid level
            std::mutex gpuMutex_;
            std::map<std::string, bool> gpuOk_;
            std::shared_ptr<ArraySource> source_;   // a step's output: read through this
            DatasetMeta sourceMeta_;
        };

    } // namespace

    // --- the service ----------------------------------------------------------------------------

    struct DatasetService::Impl {
        Options options;
        VolumeCache volumes;
        std::mutex openMutex;
        // the eight opened last, by (path, options), with the file's time stamp
        std::list<std::pair<std::string, std::pair<fs::file_time_type, std::shared_ptr<Dataset>>>> open;

        std::mutex resolverMutex;
        OutputResolver resolver;

        explicit Impl(Options o) : options(std::move(o)), volumes(cacheBudget(options.cacheBytes)) {}

        std::shared_ptr<Dataset> dataset(const json& params) {
            const std::string given = params.contains("path") && params["path"].is_string() ? params["path"].get<std::string>() : std::string();
            if (given.empty()) throw DatasetError("no path");
            if (given.rfind(kOutputPrefix, 0) == 0) {
                OutputResolver r;
                {
                    const std::lock_guard<std::mutex> g(resolverMutex);
                    r = resolver;
                }
                if (!r) throw DatasetError(given + ": this server holds no step outputs");
                return std::make_shared<Dataset>(given, r(given), volumes);
            }
            std::string expanded = given;
            if (expanded.rfind("~", 0) == 0) expanded = host::homeDirectory() + expanded.substr(1);
            std::error_code ec;
            fs::path full = fs::absolute(fs::u8path(expanded), ec);
            if (ec) full = fs::u8path(expanded);
            full = full.lexically_normal();
            const std::string path = full.u8string();
            const json opts = params.contains("options") && params["options"].is_object() ? params["options"] : json::object();
            fs::file_time_type stamp = fs::last_write_time(full, ec);
            if (ec) throw DatasetError(given + ": no such file on " + hostName());
            // a folder's own time does not move when a file in it is written anew: its manifest's does
            if (std::error_code mec; fs::is_directory(full, mec)) {
                const fs::file_time_type m = fs::last_write_time(full / DatasetManifest::kFileName, mec);
                if (!mec) stamp = m;
            }
            // The file's time is part of the key, so the volumes of a file
            // written anew are not served from the cache.
            const std::string key = path + "\n" + optionsKey(opts).text() + "\n" + std::to_string(stamp.time_since_epoch().count());
            {
                const std::lock_guard<std::mutex> g(openMutex);
                for (auto it = open.begin(); it != open.end(); ++it)
                    if (it->first == key && it->second.first == stamp) {
                        open.splice(open.begin(), open, it);
                        return it->second.second;
                    }
            }
            auto ds = std::make_shared<Dataset>(path, opts, volumes, key);
            const std::lock_guard<std::mutex> g(openMutex);
            open.remove_if([&](const auto& e) { return e.first == key; });
            open.emplace_front(key, std::make_pair(stamp, ds));
            while (open.size() > 8) open.pop_back();
            return ds;
        }

        // the request's decode device, else the service's
        std::string device(const json& params) const {
            if (params.contains("device") && params["device"].is_string()) {
                const std::string d = lowerTrim(params["device"].get<std::string>());
                if (!d.empty() && d != "auto") return d;
            }
            return options.device.empty() ? std::string("auto") : options.device;
        }
    };

    DatasetService::DatasetService() : DatasetService(Options{}) {}
    DatasetService::DatasetService(Options options) : impl_(std::make_unique<Impl>(std::move(options))) {}
    DatasetService::~DatasetService() = default;

    bool DatasetService::handles(const std::string& m) {
        return m == "dataset_info" || m == "dataset_read" || m == "dataset_view" || m == "dataset_stats";
    }

    json DatasetService::tiffReader(const std::string& device) const {
        return {{"sirius", buildInfo().version}, {"nvtiff", gpuFor(device.empty() ? impl_->options.device : device).has_value()}};
    }

    void DatasetService::setOutputResolver(OutputResolver resolver) {
        const std::lock_guard<std::mutex> g(impl_->resolverMutex);
        impl_->resolver = std::move(resolver);
    }

    void DatasetService::forgetAll() {
        impl_->volumes.clear();
        const std::lock_guard<std::mutex> g(impl_->openMutex);
        impl_->open.clear();
    }

    std::size_t DatasetService::cachedBytes() const { return impl_->volumes.used(); }

    rpc::Reply DatasetService::handle(const std::string& method, const json& params) {
        try {
            const std::shared_ptr<Dataset> ds = impl_->dataset(params);
            const std::string device = impl_->device(params);
            rpc::Reply reply;
            if (method == "dataset_info") {
                reply.result = ds->meta();
                reply.result["encodings"] = codec::availableEncodings();
                return reply;
            }
            const Index c = intParam(params, "c", 0), t = intParam(params, "t", 0);
            if (method == "dataset_stats") {
                reply.result = ds->stats(c, t, device);
                return reply;
            }
            HostArray arr;
            if (method == "dataset_read") {
                if (params.contains("z") && !params["z"].is_null()) arr = ds->plane(c, t, intOf(params["z"], 0), device);
                else arr = *ds->volume(c, t, device);
            } else if (method == "dataset_view") {
                const std::string kind = params.contains("kind") && params["kind"].is_string() ? params["kind"].get<std::string>() : std::string("xy");
                arr = ds->view(kind, c, t, intParam(params, "index", 0, true), intParam(params, "factor", 1, true),
                               params.contains("region") ? params["region"] : json(nullptr), intParam(params, "max_side", 256, true), device);
            } else {
                throw DatasetError("unknown method '" + method + "'");
            }
            if (arr.dtype == "bool") {
                // not a protocol dtype: float32, as the worker sends it
                const std::vector<float> f = toFloat(arr);
                arr.dtype = "float32";
                arr.bytes.resize(f.size() * sizeof(float));
                if (!f.empty()) std::memcpy(arr.bytes.data(), f.data(), arr.bytes.size());
            }
            std::vector<std::string> accept;
            if (params.contains("accept") && params["accept"].is_array())
                for (const json& a : params["accept"])
                    if (a.is_string()) accept.push_back(a.get<std::string>());
            codec::Encoded e = codec::encodeArray(arr.dtype, arr.shape, std::move(arr.bytes), accept, "data");
            reply.result = std::move(e.description);
            reply.tensors.push_back(std::move(e.tensor));
            return reply;
        } catch (const DatasetError& e) {
            throw std::runtime_error(std::string("DatasetError: ") + e.what());
        }
    }

} // namespace sirius::app
