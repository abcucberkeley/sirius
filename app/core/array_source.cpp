#include "core/array_source.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <sirius/tiff_io.hpp>
#include <sirius/zarr_io.hpp>

#include "core/manifest.hpp"

namespace sirius::app {

    namespace fs = std::filesystem;

    // --- PageOrder ------------------------------------------------------------

    Index PageOrder::planeOf(Index ci, Index ti, Index zi) const noexcept {
        // The order string lists the fastest varying axis first.
        Index page = 0, stride = 1;
        for (char a : order) {
            switch (a) {
                case 'c':
                    page += ci * stride;
                    stride *= std::max<Index>(c, 1);
                    break;
                case 't':
                    page += ti * stride;
                    stride *= std::max<Index>(t, 1);
                    break;
                case 'z':
                    page += zi * stride;
                    stride *= std::max<Index>(z, 1);
                    break;
                default: break;
            }
        }
        return page;
    }

    PageOrder PageOrder::fromDims(const Dims5& d, const std::string& order) {
        PageOrder p;
        p.order = order;
        p.c = d.c;
        p.t = d.t;
        p.z = d.z;
        return p;
    }

    namespace {
        // "czt" with every letter exactly once; anything else -> "czt".
        std::string normalizeOrder(std::string order) {
            std::string out;
            for (char ch : order) {
                const char l = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                if ((l == 'c' || l == 't' || l == 'z') && out.find(l) == std::string::npos) out += l;
            }
            for (char l : std::string("czt"))
                if (out.find(l) == std::string::npos) out += l;
            return out;
        }

        // OME DimensionOrder "XYCZT" -> page order "czt" (after the XY plane, C
        // varies fastest, then Z, then T).
        std::string pageOrderFromOme(const std::string& dimensionOrder) {
            std::string out;
            for (char ch : dimensionOrder) {
                const char l = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                if (l == 'c' || l == 'z' || l == 't') out += l;
            }
            return normalizeOrder(out);
        }
    } // namespace

    // --- ArraySource defaults ----------------------------------------------------

    bool isFolderDataset(const std::string& path) {
        std::error_code ec;
        return std::filesystem::is_directory(path, ec) && std::filesystem::exists(std::filesystem::path(path) / "sirius-dataset.toml", ec);
    }

    void ArraySource::readTileVolume(Index tile, Index c, Index t, float* out) const {
        if (tile != currentTile())
            throw std::out_of_range("this dataset serves one tile at a time; select tile " + std::to_string(tile) + " first");
        readVolume(c, t, out);
    }

    void ArraySource::readVolume(Index c, Index t, float* out) const {
        const Dims5& d = dims();
        for (Index z = 0; z < d.z; ++z) readPlane(c, t, z, out + z * d.planeSize());
    }

    std::shared_ptr<Array5> ArraySource::readAll(const ProgressFn& progress) const {
        const Dims5& d = dims();
        auto out = std::make_shared<Array5>(d);
        const Index total = d.c * d.t;
        Index done = 0;
        for (Index c = 0; c < d.c; ++c)
            for (Index t = 0; t < d.t; ++t) {
                readVolume(c, t, out->plane(c, t, 0));
                ++done;
                if (progress) progress(static_cast<double>(done) / static_cast<double>(total), "reading");
            }
        return out;
    }

    // --- MemorySource --------------------------------------------------------------

    MemorySource::MemorySource(ArrayPtr array, DatasetMeta meta) : array_(std::move(array)), meta_(std::move(meta)) {
        if (!array_) throw std::invalid_argument("MemorySource: null array");
        meta_.dims = array_->dims();
        if (meta_.format.empty()) meta_.format = "memory";
        meta_.normalizeChannels();
    }

    void MemorySource::readPlane(Index c, Index t, Index z, float* out) const {
        std::memcpy(out, array_->plane(c, t, z), static_cast<std::size_t>(array_->dims().planeSize()) * sizeof(float));
    }

    void MemorySource::readVolume(Index c, Index t, float* out) const {
        std::memcpy(out, array_->plane(c, t, 0),
                    static_cast<std::size_t>(array_->dims().z * array_->dims().planeSize()) * sizeof(float));
    }

    std::shared_ptr<Array5> MemorySource::readAll(const ProgressFn& progress) const {
        if (progress) progress(1.0, "in memory");
        // Shared ownership of the const array: callers never mutate a source's data.
        return std::const_pointer_cast<Array5>(array_);
    }

    // --- TIFF metadata ---------------------------------------------------------------

    namespace {

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        std::string trim(const std::string& s) {
            const auto a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
            return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
        }

        // Length unit -> micrometres.
        double unitToUm(const std::string& unitIn) {
            const std::string u = lower(trim(unitIn));
            if (u.empty() || u == "µm" || u == "\xc2\xb5m" || u == "um" || u == "micron" || u == "microns" || u == "micrometer" ||
                u == "micrometre" || u == "\xce\xbcm")
                return 1.0;
            if (u == "nm" || u == "nanometer") return 1e-3;
            if (u == "mm" || u == "millimeter") return 1e3;
            if (u == "cm" || u == "centimeter") return 1e4;
            if (u == "m" || u == "meter") return 1e6;
            if (u == "inch" || u == "in") return 2.54e4;
            if (u == "pixel" || u == "pixels") return 0.0;
            return 1.0;
        }

        double timeUnitToS(const std::string& unitIn) {
            const std::string u = lower(trim(unitIn));
            if (u.empty() || u == "s" || u == "sec" || u == "second" || u == "seconds") return 1.0;
            if (u == "ms") return 1e-3;
            if (u == "us" || u == "µs") return 1e-6;
            if (u == "min") return 60.0;
            if (u == "h") return 3600.0;
            return 1.0;
        }

        // Decode the handful of XML entities OME-XML attribute values use.
        std::string xmlUnescape(std::string s) {
            struct E {
                const char* from;
                const char* to;
            };
            static const E ents[] = {{"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}};
            for (const E& e : ents) {
                std::size_t pos = 0;
                while ((pos = s.find(e.from, pos)) != std::string::npos) {
                    s.replace(pos, std::strlen(e.from), e.to);
                    pos += std::strlen(e.to);
                }
            }
            // &#181; style numeric references (µ appears as &#181; in some writers)
            std::size_t pos = 0;
            while ((pos = s.find("&#", pos)) != std::string::npos) {
                const std::size_t end = s.find(';', pos);
                if (end == std::string::npos) break;
                const std::string num = s.substr(pos + 2, end - pos - 2);
                unsigned long code = 0;
                try {
                    code = num.size() > 1 && (num[0] == 'x' || num[0] == 'X') ? std::stoul(num.substr(1), nullptr, 16) : std::stoul(num);
                } catch (...) {
                    pos = end + 1;
                    continue;
                }
                std::string utf8;
                if (code < 0x80) utf8 += static_cast<char>(code);
                else if (code < 0x800) {
                    utf8 += static_cast<char>(0xC0 | (code >> 6));
                    utf8 += static_cast<char>(0x80 | (code & 0x3F));
                } else {
                    utf8 += static_cast<char>(0xE0 | (code >> 12));
                    utf8 += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                    utf8 += static_cast<char>(0x80 | (code & 0x3F));
                }
                s.replace(pos, end - pos + 1, utf8);
                pos += utf8.size();
            }
            return s;
        }

        // Attributes of one XML start tag (the text between '<Name' and '>').
        using Attrs = std::map<std::string, std::string>;

        Attrs parseAttrs(const std::string& tag) {
            Attrs a;
            std::size_t i = 0;
            while (i < tag.size()) {
                while (i < tag.size() && (std::isspace(static_cast<unsigned char>(tag[i])) || tag[i] == '/')) ++i;
                const std::size_t nameStart = i;
                while (i < tag.size() && tag[i] != '=' && !std::isspace(static_cast<unsigned char>(tag[i]))) ++i;
                if (i >= tag.size()) break;
                const std::string name = tag.substr(nameStart, i - nameStart);
                while (i < tag.size() && (std::isspace(static_cast<unsigned char>(tag[i])) || tag[i] == '=')) ++i;
                if (i >= tag.size()) break;
                const char quote = tag[i];
                if (quote != '"' && quote != '\'') {
                    ++i;
                    continue;
                }
                const std::size_t valueStart = ++i;
                const std::size_t valueEnd = tag.find(quote, valueStart);
                if (valueEnd == std::string::npos) break;
                a[name] = xmlUnescape(tag.substr(valueStart, valueEnd - valueStart));
                i = valueEnd + 1;
            }
            return a;
        }

        // Every start tag named `name` (with or without a namespace prefix)
        // inside `xml`, as attribute maps, in document order.
        std::vector<Attrs> findTags(const std::string& xml, const std::string& name) {
            std::vector<Attrs> out;
            std::size_t pos = 0;
            while ((pos = xml.find('<', pos)) != std::string::npos) {
                std::size_t i = pos + 1;
                if (i < xml.size() && (xml[i] == '/' || xml[i] == '?' || xml[i] == '!')) {
                    ++pos;
                    continue;
                }
                std::size_t nameEnd = i;
                while (nameEnd < xml.size() && !std::isspace(static_cast<unsigned char>(xml[nameEnd])) && xml[nameEnd] != '>' && xml[nameEnd] != '/')
                    ++nameEnd;
                std::string tagName = xml.substr(i, nameEnd - i);
                const std::size_t colon = tagName.find(':');
                if (colon != std::string::npos) tagName = tagName.substr(colon + 1);
                const std::size_t close = xml.find('>', nameEnd);
                if (close == std::string::npos) break;
                if (tagName == name) out.push_back(parseAttrs(xml.substr(nameEnd, close - nameEnd)));
                pos = close + 1;
            }
            return out;
        }

        double attrDouble(const Attrs& a, const char* key, double def = 0.0) {
            auto it = a.find(key);
            if (it == a.end()) return def;
            try {
                return std::stod(it->second);
            } catch (...) { return def; }
        }
        std::string attrString(const Attrs& a, const char* key) {
            auto it = a.find(key);
            return it == a.end() ? std::string() : it->second;
        }

        // OME Color: signed 32-bit RGBA (r << 24 | g << 16 | b << 8 | a).
        std::array<float, 3> omeColor(const std::string& s, bool& ok) {
            ok = false;
            if (s.empty()) return {1.f, 1.f, 1.f};
            long long v = 0;
            try {
                v = std::stoll(s);
            } catch (...) { return {1.f, 1.f, 1.f}; }
            const std::uint32_t u = static_cast<std::uint32_t>(static_cast<std::int32_t>(v));
            ok = true;
            return {static_cast<float>((u >> 24) & 0xFF) / 255.f, static_cast<float>((u >> 16) & 0xFF) / 255.f,
                    static_cast<float>((u >> 8) & 0xFF) / 255.f};
        }

        void parseOme(const std::string& xml, ParsedTiffMetadata& m) {
            m.ome = true;
            const auto pixels = findTags(xml, "Pixels");
            if (!pixels.empty()) {
                const Attrs& p = pixels.front();
                m.c = static_cast<Index>(attrDouble(p, "SizeC", 0));
                m.t = static_cast<Index>(attrDouble(p, "SizeT", 0));
                m.z = static_cast<Index>(attrDouble(p, "SizeZ", 0));
                m.dimensionOrder = attrString(p, "DimensionOrder");
                const double ux = unitToUm(attrString(p, "PhysicalSizeXUnit"));
                const double uy = unitToUm(attrString(p, "PhysicalSizeYUnit"));
                const double uz = unitToUm(attrString(p, "PhysicalSizeZUnit"));
                m.voxelUm = {attrDouble(p, "PhysicalSizeX") * ux, attrDouble(p, "PhysicalSizeY") * uy,
                             attrDouble(p, "PhysicalSizeZ") * uz};
                m.frameIntervalS = attrDouble(p, "TimeIncrement") * timeUnitToS(attrString(p, "TimeIncrementUnit"));
            }
            for (const Attrs& c : findTags(xml, "Channel")) {
                ChannelInfo ch;
                ch.label = attrString(c, "Name");
                const double em = attrDouble(c, "EmissionWavelength");
                const double emUnit = unitToUm(attrString(c, "EmissionWavelengthUnit").empty() ? "nm" : attrString(c, "EmissionWavelengthUnit"));
                ch.wavelengthNm = em > 0 ? em * emUnit * 1e3 : 0.0;   // um -> nm
                bool ok = false;
                const auto color = omeColor(attrString(c, "Color"), ok);
                if (ok && !(color[0] == 1.f && color[1] == 1.f && color[2] == 1.f)) ch.color = color;
                m.channels.push_back(std::move(ch));
            }
        }

        void parseImageJ(const std::string& text, ParsedTiffMetadata& m) {
            m.imagej = true;
            std::map<std::string, std::string> kv;
            std::istringstream in(text);
            std::string line;
            while (std::getline(in, line)) {
                const std::size_t eq = line.find('=');
                if (eq == std::string::npos) continue;
                kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
            }
            auto num = [&](const char* key, double def) {
                auto it = kv.find(key);
                if (it == kv.end()) return def;
                try {
                    return std::stod(it->second);
                } catch (...) { return def; }
            };
            m.c = static_cast<Index>(num("channels", 0));
            m.z = static_cast<Index>(num("slices", 0));
            m.t = static_cast<Index>(num("frames", 0));
            m.dimensionOrder = "XYCZT";   // ImageJ hyperstacks: channel fastest, then slice, then frame
            const double unit = unitToUm(kv.count("unit") ? kv["unit"] : "");
            const double spacing = num("spacing", 0.0);
            if (spacing > 0 && unit > 0) m.voxelUm[2] = spacing * unit;
            m.frameIntervalS = num("finterval", 0.0);
            // the x/y pixel size comes from the resolution tags; remember the unit
            if (unit > 0) m.voxelUm[0] = m.voxelUm[1] = -unit;   // marker: multiply by 1/resolution
        }

    } // namespace

    ParsedTiffMetadata parseTiffDescription(const std::string& description) {
        ParsedTiffMetadata m;
        if (description.find("<OME") != std::string::npos || description.find("<ome") != std::string::npos) parseOme(description, m);
        else if (description.rfind("ImageJ=", 0) == 0 || description.find("\nImageJ=") != std::string::npos) parseImageJ(description, m);
        return m;
    }

    // --- TIFF source -------------------------------------------------------------------

    namespace {

        class TiffArraySource final : public ArraySource {
        public:
            TiffArraySource(std::string path, DatasetMeta meta, PageOrder order)
                : file_(std::move(path)), meta_(std::move(meta)), order_(order) {
                const Dims5& d = meta_.dims;
                // cache: up to 64 planes or 512 MB, whichever is smaller
                const std::size_t planeBytes = static_cast<std::size_t>(d.planeSize()) * sizeof(float);
                capacity_ = std::max<std::size_t>(2, std::min<std::size_t>(64, (std::size_t{512} << 20) / std::max<std::size_t>(planeBytes, 1)));
                zFastest_ = order_.order.rfind('z', 0) == 0;
                contiguousStack_ = order_.order == "ztc";
            }

            const DatasetMeta& meta() const noexcept override { return meta_; }
            bool gpuDecodable() const noexcept override {
                try {
                    return file_.gpuDecodable();
                } catch (...) { return false; }
            }

            void readPlane(Index c, Index t, Index z, float* out) const override {
                const Dims5& d = meta_.dims;
                check(c, t, z);
                const Index page = order_.planeOf(c, t, z);
                const std::size_t n = static_cast<std::size_t>(d.planeSize());
                {
                    std::lock_guard<std::mutex> g(mutex_);
                    auto it = index_.find(page);
                    if (it != index_.end()) {
                        std::memcpy(out, it->second->data.data(), n * sizeof(float));
                        lru_.splice(lru_.begin(), lru_, it->second);   // most recently used
                        return;
                    }
                }
                Buffer<float> plane = file_.readPages<float>(static_cast<std::size_t>(page), 1);
                std::memcpy(out, plane.data(), n * sizeof(float));
                std::lock_guard<std::mutex> g(mutex_);
                if (index_.count(page)) return;
                lru_.push_front(Entry{page, std::vector<float>(plane.data(), plane.data() + n)});
                index_[page] = lru_.begin();
                while (lru_.size() > capacity_) {
                    index_.erase(lru_.back().page);
                    lru_.pop_back();
                }
            }

            void readVolume(Index c, Index t, float* out) const override {
                const Dims5& d = meta_.dims;
                check(c, t, 0);
                if (zFastest_) {
                    // the z planes of one (c, t) are consecutive pages
                    const Index first = order_.planeOf(c, t, 0);
                    Buffer<float> vol = file_.readPages<float>(static_cast<std::size_t>(first), static_cast<std::size_t>(d.z));
                    std::memcpy(out, vol.data(), static_cast<std::size_t>(d.z * d.planeSize()) * sizeof(float));
                    return;
                }
                ArraySource::readVolume(c, t, out);
            }

            std::shared_ptr<Array5> readAll(const ProgressFn& progress) const override {
                const Dims5& d = meta_.dims;
                if (progress) progress(0.0, "reading " + fs::path(file_.path()).filename().string());
                Buffer<float> stack = file_.readStack<float>();
                if (contiguousStack_) {
                    auto out = std::make_shared<Array5>(Array5::fromBuffer(std::move(stack), d));
                    if (progress) progress(1.0, "read");
                    return out;
                }
                auto out = std::make_shared<Array5>(d);
                const std::size_t n = static_cast<std::size_t>(d.planeSize());
                for (Index c = 0; c < d.c; ++c)
                    for (Index t = 0; t < d.t; ++t)
                        for (Index z = 0; z < d.z; ++z)
                            std::memcpy(out->plane(c, t, z), stack.data() + order_.planeOf(c, t, z) * d.planeSize(), n * sizeof(float));
                if (progress) progress(1.0, "read");
                return out;
            }

        private:
            struct Entry {
                Index page;
                std::vector<float> data;
            };
            void check(Index c, Index t, Index z) const {
                const Dims5& d = meta_.dims;
                if (c < 0 || c >= d.c || t < 0 || t >= d.t || z < 0 || z >= d.z)
                    throw std::out_of_range("plane (c " + std::to_string(c) + ", t " + std::to_string(t) + ", z " +
                                            std::to_string(z) + ") outside " + d.toString());
            }

            TiffFile file_;
            DatasetMeta meta_;
            PageOrder order_;
            bool zFastest_ = false;
            bool contiguousStack_ = false;
            std::size_t capacity_ = 16;
            mutable std::mutex mutex_;
            mutable std::list<Entry> lru_;
            mutable std::unordered_map<Index, std::list<Entry>::iterator> index_;
        };

        bool looksLikeTiff(const std::string& path) {
            const std::string ext = lower(fs::path(path).extension().string());
            return ext == ".tif" || ext == ".tiff" || ext == ".btf" || ext == ".tf8" || ext == ".ome";
        }

        // Voxel size from the resolution tags (pixels per unit).
        std::array<double, 2> pixelFromResolution(const TiffImageInfo& p, double imagejUnitUm) {
            std::array<double, 2> out{0.0, 0.0};
            if (p.xResolution <= 0.0 || p.yResolution <= 0.0) return out;
            // The ResolutionUnit tag wins when it names a real unit; ImageJ
            // writes "none" and puts the unit in its description instead.
            double unitUm = 0.0;
            if (p.resolutionUnit == 3) unitUm = 1e4;             // cm
            else if (p.resolutionUnit == 2) unitUm = 2.54e4;     // inch
            else if (imagejUnitUm > 0.0) unitUm = imagejUnitUm;
            if (unitUm <= 0.0) return out;
            out = {unitUm / p.xResolution, unitUm / p.yResolution};
            // sanity: microscopy pixels are 1 nm .. 100 um
            for (double& v : out)
                if (!(v > 1e-3 && v < 100.0)) v = 0.0;
            return out;
        }

        struct TiffProbe {
            DatasetMeta meta;
            PageOrder order;
            std::string summary;
            bool dimsFromMetadata = false;
        };

        TiffProbe probeTiff(const std::string& path, const OpenOptions* options) {
            const TiffInfo info = inspectTiff(path);
            if (!info.uniformPages()) throw std::runtime_error("TIFF pages differ in size or pixel type: " + path);
            const TiffImageInfo& p0 = info.page(0);
            const Index pages = static_cast<Index>(info.pageCount());

            TiffProbe r;
            DatasetMeta& m = r.meta;
            m.name = fs::path(path).stem().string();
            if (fs::path(m.name).extension() == ".ome") m.name = fs::path(m.name).stem().string();
            m.sourcePath = path;
            m.sourceType = p0.pixelType;
            std::error_code ec;
            m.bytesOnDisk = fs::exists(path, ec) ? static_cast<std::uint64_t>(fs::file_size(path, ec)) : 0;
            m.dims.y = static_cast<Index>(p0.height);
            m.dims.x = static_cast<Index>(p0.width);

            const ParsedTiffMetadata md = parseTiffDescription(p0.description);
            m.format = md.ome ? "ome-tiff" : "tiff";
            std::ostringstream summary;
            summary << (md.ome ? "OME-TIFF" : md.imagej ? "ImageJ TIFF"
                                                        : "TIFF")
                    << " · " << pages << (pages == 1 ? " page" : " pages")
                    << " · " << toString(p0.pixelType);

            // dimensions: explicit page order > OME / ImageJ metadata > pages as z
            Index c = 1, t = 1, z = pages;
            std::string order = "czt";
            const bool described = (md.ome || md.imagej) && (md.c > 0 || md.t > 0 || md.z > 0);
            if (options && options->pageOrder) {
                // An axis left at 0 is not "1": it is whatever the file says
                // (a 2-channel OME stack with only z given stays 2 channels),
                // and the page order likewise unless one was given.
                const PageOrder& po = *options->pageOrder;
                c = po.c > 0 ? po.c : (described && md.c > 0 ? md.c : 1);
                t = po.t > 0 ? po.t : (described && md.t > 0 ? md.t : 1);
                z = po.z > 0 ? po.z : std::max<Index>(pages / std::max<Index>(c * t, 1), 1);
                order = described && po.order == "czt" && !md.dimensionOrder.empty() ? pageOrderFromOme(md.dimensionOrder)
                                                                                     : normalizeOrder(po.order);
                r.dimsFromMetadata = described && po.c <= 0 && po.t <= 0 && po.z <= 0;
            } else if (described) {
                c = std::max<Index>(md.c, 1);
                t = std::max<Index>(md.t, 1);
                z = md.z > 0 ? md.z : std::max<Index>(pages / (c * t), 1);
                order = pageOrderFromOme(md.dimensionOrder);
                r.dimsFromMetadata = true;
            }
            if (c * t * z != pages) {
                summary << " · metadata says c" << c << " t" << t << " z" << z << " (" << c * t * z << " planes) but the file has " << pages
                        << " pages: reading pages as z";
                c = 1;
                t = 1;
                z = pages;
                order = "czt";
                r.dimsFromMetadata = false;
            }
            m.dims.c = c;
            m.dims.t = t;
            m.dims.z = z;
            r.order = PageOrder::fromDims(m.dims, order);

            // voxel size
            std::array<double, 3> voxel{0.0, 0.0, 0.0};
            if (md.ome) voxel = md.voxelUm;
            const double imagejUnit = md.imagej && md.voxelUm[0] < 0 ? -md.voxelUm[0] : 0.0;
            if (voxel[0] <= 0.0 || voxel[1] <= 0.0) {
                const auto xy = pixelFromResolution(p0, imagejUnit);
                if (xy[0] > 0.0) voxel[0] = xy[0];
                if (xy[1] > 0.0) voxel[1] = xy[1];
            }
            if (voxel[2] <= 0.0 && md.imagej) voxel[2] = md.voxelUm[2];
            if (options && options->voxelUm)
                for (int k = 0; k < 3; ++k)
                    if ((*options->voxelUm)[k] > 0.0) voxel[k] = (*options->voxelUm)[k];   // 0 = the file's
            const bool knownXy = voxel[0] > 0.0 && voxel[1] > 0.0;
            if (!knownXy) voxel[0] = voxel[1] = 0.1;
            if (voxel[2] <= 0.0) voxel[2] = knownXy ? voxel[0] * 2.0 : 0.2;
            m.voxelUm = voxel;
            if (knownXy) summary << " · voxel " << m.voxelString();
            m.frameIntervalS = md.frameIntervalS;

            // channels
            if (options && options->channels) m.channels = *options->channels;
            else if (md.ome && !md.channels.empty()) m.channels = md.channels;
            m.normalizeChannels();
            if (m.dims.c > 1) summary << " · " << m.dims.c << " channels";

            if (options && options->sim) m.sim = *options->sim;
            if (m.sim.present) m.acquisition = "3D-SIM raw · " + std::to_string(m.sim.sectionsPerPlane()) + " phase images per plane";
            else if (md.ome) m.acquisition = "OME-TIFF";
            else if (md.imagej) m.acquisition = "ImageJ hyperstack";
            r.summary = summary.str();
            return r;
        }

        // --- zarr source ------------------------------------------------------------

        // Map a store's axes onto (c, t, z, y, x). Named axes win; unnamed ones
        // are assigned by position from the fast end (x, y, z, c, t).
        struct AxisMap {
            std::vector<Axis> axisOf;           // per store dimension
            std::array<int, 5> dimOf{-1, -1, -1, -1, -1};   // store dimension per Axis, -1 = absent (extent 1)
        };

        AxisMap mapAxes(const std::vector<std::string>& names, int rank) {
            AxisMap m;
            m.axisOf.resize(static_cast<std::size_t>(rank), Axis::X);
            std::array<bool, 5> used{false, false, false, false, false};
            std::vector<bool> assigned(static_cast<std::size_t>(rank), false);
            for (int i = 0; i < rank; ++i) {
                if (i >= static_cast<int>(names.size())) break;
                const std::string n = lower(names[static_cast<std::size_t>(i)]);
                std::optional<Axis> a;
                if (n == "c" || n == "channel") a = Axis::C;
                else if (n == "t" || n == "time") a = Axis::T;
                else if (n == "z") a = Axis::Z;
                else if (n == "y") a = Axis::Y;
                else if (n == "x") a = Axis::X;
                if (a && !used[static_cast<std::size_t>(*a)]) {
                    m.axisOf[static_cast<std::size_t>(i)] = *a;
                    used[static_cast<std::size_t>(*a)] = true;
                    assigned[static_cast<std::size_t>(i)] = true;
                }
            }
            static const Axis fromFastEnd[] = {Axis::X, Axis::Y, Axis::Z, Axis::C, Axis::T};
            std::size_t next = 0;
            for (int i = rank; i-- > 0;) {
                if (assigned[static_cast<std::size_t>(i)]) continue;
                while (next < 5 && used[static_cast<std::size_t>(fromFastEnd[next])]) ++next;
                if (next >= 5) throw std::runtime_error("zarr store has more than five dimensions that could be mapped");
                m.axisOf[static_cast<std::size_t>(i)] = fromFastEnd[next];
                used[static_cast<std::size_t>(fromFastEnd[next])] = true;
            }
            for (int i = 0; i < rank; ++i) m.dimOf[static_cast<std::size_t>(m.axisOf[static_cast<std::size_t>(i)])] = i;
            return m;
        }

        class ZarrArraySource final : public ArraySource {
        public:
            ZarrArraySource(std::string path, DatasetMeta meta, AxisMap map)
                : array_(path), meta_(std::move(meta)), map_(std::move(map)) {}

            const DatasetMeta& meta() const noexcept override { return meta_; }

            void readPlane(Index c, Index t, Index z, float* out) const override {
                const Dims5& d = meta_.dims;
                if (c < 0 || c >= d.c || t < 0 || t >= d.t || z < 0 || z >= d.z)
                    throw std::out_of_range("plane (c " + std::to_string(c) + ", t " + std::to_string(t) + ", z " +
                                            std::to_string(z) + ") outside " + d.toString());
                const int rank = array_.info().rank();
                std::vector<Index> origin(static_cast<std::size_t>(rank), 0), shape(static_cast<std::size_t>(rank), 1);
                auto set = [&](Axis a, Index o, Index n) {
                    const int dim = map_.dimOf[static_cast<std::size_t>(a)];
                    if (dim < 0) return;
                    origin[static_cast<std::size_t>(dim)] = o;
                    shape[static_cast<std::size_t>(dim)] = n;
                };
                set(Axis::C, c, 1);
                set(Axis::T, t, 1);
                set(Axis::Z, z, 1);
                set(Axis::Y, 0, d.y);
                set(Axis::X, 0, d.x);
                if (yxLast_) {
                    array_.read<float>(origin, shape, out);
                } else {
                    // y before x is not guaranteed: read then transpose
                    Buffer<float> tmp = array_.read<float>(origin, shape);
                    const int dy = map_.dimOf[static_cast<std::size_t>(Axis::Y)], dx = map_.dimOf[static_cast<std::size_t>(Axis::X)];
                    if (dy > dx) {   // stored x-major: tmp is (x, y)
                        for (Index y = 0; y < d.y; ++y)
                            for (Index x = 0; x < d.x; ++x) out[y * d.x + x] = tmp.data()[x * d.y + y];
                    } else {
                        std::memcpy(out, tmp.data(), static_cast<std::size_t>(d.planeSize()) * sizeof(float));
                    }
                }
            }

            void setYxLast(bool v) { yxLast_ = v; }

        private:
            ZarrArray array_;
            DatasetMeta meta_;
            AxisMap map_;
            bool yxLast_ = true;
        };

        struct ZarrProbe {
            DatasetMeta meta;
            AxisMap map;
            std::string summary;
            bool yxLast = true;
        };

        ZarrProbe probeZarr(const std::string& path, const OpenOptions* options) {
            if (!zarrSupported()) throw std::runtime_error("this build cannot open zarr / N5 stores (SIRIUS_ENABLE_TENSORSTORE=OFF): " + path);
            const ZarrArrayInfo info = inspectZarr(path);
            ZarrProbe r;
            DatasetMeta& m = r.meta;
            m.name = fs::path(path).filename().string();
            if (m.name.empty()) m.name = fs::path(path).parent_path().filename().string();
            m.name = fs::path(m.name).stem().string();
            m.sourcePath = path;
            m.format = info.driver == "n5" ? "n5" : "zarr";
            m.sourceType = info.pixelType;
            m.bytesOnDisk = info.bytesOnDisk;
            const int rank = info.rank();
            if (rank < 2 || rank > 5) throw std::runtime_error("zarr store has rank " + std::to_string(rank) + "; 2..5 supported");
            r.map = mapAxes(info.axes, rank);
            for (int i = 0; i < rank; ++i) m.dims[r.map.axisOf[static_cast<std::size_t>(i)]] = info.shape[static_cast<std::size_t>(i)];
            const int dy = r.map.dimOf[static_cast<std::size_t>(Axis::Y)], dx = r.map.dimOf[static_cast<std::size_t>(Axis::X)];
            r.yxLast = dy == rank - 2 && dx == rank - 1;

            std::array<double, 3> voxel{0.0, 0.0, 0.0};
            if (info.scale.size() == static_cast<std::size_t>(rank)) {
                auto sc = [&](Axis a) {
                    const int dim = r.map.dimOf[static_cast<std::size_t>(a)];
                    return dim < 0 ? 0.0 : info.scale[static_cast<std::size_t>(dim)];
                };
                voxel = {sc(Axis::X), sc(Axis::Y), sc(Axis::Z)};
            }
            if (options && options->voxelUm) voxel = *options->voxelUm;
            const bool known = voxel[0] > 0.0 && voxel[1] > 0.0;
            if (!known) voxel[0] = voxel[1] = 0.1;
            if (voxel[2] <= 0.0) voxel[2] = known ? voxel[0] * 2.0 : 0.2;
            m.voxelUm = voxel;
            if (info.scale.size() == static_cast<std::size_t>(rank) && r.map.dimOf[static_cast<std::size_t>(Axis::T)] >= 0)
                m.frameIntervalS = info.scale[static_cast<std::size_t>(r.map.dimOf[static_cast<std::size_t>(Axis::T)])];

            if (options && options->channels) m.channels = *options->channels;
            else {
                for (std::size_t i = 0; i < info.channelNames.size(); ++i) {
                    ChannelInfo ch;
                    ch.label = info.channelNames[i];
                    if (i < info.channelColors.size() && info.channelColors[i].size() == 7) {
                        try {
                            ch.color = colorFromHex(info.channelColors[i]);
                        } catch (...) {}
                    }
                    m.channels.push_back(std::move(ch));
                }
            }
            m.normalizeChannels();
            if (options && options->sim) m.sim = *options->sim;
            m.acquisition = info.multiscalePaths.size() > 1 ? "OME-Zarr · " + std::to_string(info.multiscalePaths.size()) + " resolution levels"
                                                            : (info.isGroup ? "OME-Zarr" : m.format);
            std::ostringstream s;
            s << (info.driver == "n5" ? "N5" : info.driver == "zarr3" ? "zarr v3"
                                                                      : "zarr v2")
              << " · " << toString(info.pixelType)
              << " · chunks";
            for (Index c : info.chunks) s << " " << c;
            s << " · " << info.codec;
            if (m.dims.c > 1) s << " · " << m.dims.c << " channels";
            if (known) s << " · voxel " << m.voxelString();
            r.summary = s.str();
            return r;
        }

        // --- multi-file folder (manifest) ---------------------------------------------

        // One TIFF stack per (tile, channel, t), mapped by the manifest. Files
        // open on demand and a handful stay open, because a viewer scrubbing z
        // hits the same file again and again; pages convert to float32.
        class FolderArraySource final : public ArraySource {
        public:
            FolderArraySource(fs::path folder, const DatasetManifest& manifest, DatasetMeta meta)
                : folder_(std::move(folder)), meta_(std::move(meta)) {
                // resolve every (tile, c, t) once: file() is a linear scan
                tiles_ = static_cast<Index>(manifest.tiles.size());
                const Dims5& d = meta_.dims;
                paths_.assign(static_cast<std::size_t>(tiles_ * d.c * d.t), std::string());
                for (const ManifestFile& f : manifest.files) {
                    const Index k = manifest.tileIndex(f.tile), c = manifest.channelIndex(f.channel);
                    if (k < 0 || c < 0 || f.t < 0 || f.t >= d.t || c >= d.c) continue;   // validate() reported these
                    paths_[slot(k, c, f.t)] = manifestFilePath(folder_, f).string();
                }
            }

            const DatasetMeta& meta() const noexcept override { return meta_; }
            Index tileCount() const noexcept override { return tiles_; }
            Index currentTile() const noexcept override { return meta_.tileIndex; }
            void selectTile(Index tile) override {
                checkTile(tile);
                meta_.tileIndex = tile;
            }

            void readPlane(Index c, Index t, Index z, float* out) const override {
                check(c, t, z);
                const std::string& path = pathOf(meta_.tileIndex, c, t);
                std::shared_ptr<TiffFile> file = open(path);
                Buffer<float> plane = file->readPages<float>(static_cast<std::size_t>(z), 1);
                copyOut(plane, meta_.dims.planeSize(), out, path);
            }

            void readVolume(Index c, Index t, float* out) const override { readTileVolume(meta_.tileIndex, c, t, out); }

            void readTileVolume(Index tile, Index c, Index t, float* out) const override {
                checkTile(tile);
                check(c, t, 0);
                const std::string& path = pathOf(tile, c, t);
                std::shared_ptr<TiffFile> file = open(path);
                Buffer<float> volume = file->readStack<float>();
                copyOut(volume, meta_.dims.z * meta_.dims.planeSize(), out, path);
            }

            std::shared_ptr<Array5> readAll(const ProgressFn& progress) const override {
                const Dims5& d = meta_.dims;
                auto out = std::make_shared<Array5>(d);
                const double total = static_cast<double>(std::max<Index>(1, d.c * d.t));
                Index done = 0;
                for (Index c = 0; c < d.c; ++c)
                    for (Index t = 0; t < d.t; ++t) {
                        if (progress) progress(done / total, "reading " + fs::path(pathOf(meta_.tileIndex, c, t)).filename().string());
                        readTileVolume(meta_.tileIndex, c, t, out->plane(c, t, 0));
                        ++done;
                    }
                if (progress) progress(1.0, "read");
                return out;
            }

        private:
            static constexpr std::size_t kOpenFiles = 8;

            std::size_t slot(Index tile, Index c, Index t) const noexcept {
                const Dims5& d = meta_.dims;
                return static_cast<std::size_t>((tile * d.c + c) * d.t + t);
            }
            void checkTile(Index tile) const {
                if (tile < 0 || tile >= tiles_)
                    throw std::out_of_range("tile " + std::to_string(tile) + " of " + std::to_string(tiles_) + " in " + folder_.string());
            }
            void check(Index c, Index t, Index z) const {
                const Dims5& d = meta_.dims;
                if (c < 0 || c >= d.c || t < 0 || t >= d.t || z < 0 || z >= d.z)
                    throw std::out_of_range("plane (c " + std::to_string(c) + ", t " + std::to_string(t) + ", z " +
                                            std::to_string(z) + ") outside " + d.toString());
            }
            const std::string& pathOf(Index tile, Index c, Index t) const {
                const std::string& p = paths_[slot(tile, c, t)];
                if (p.empty())
                    throw std::runtime_error("no file for tile " + std::to_string(tile) + ", channel " + std::to_string(c) +
                                             ", t " + std::to_string(t) + " in " + folder_.string());
                return p;
            }
            // The file may have changed since the manifest was written: its
            // size must still be the manifest's (z, y, x).
            static void copyOut(const Buffer<float>& b, Index n, float* out, const std::string& path) {
                if (b.shape().numel() != n)
                    throw std::runtime_error(path + ": expected " + std::to_string(n) + " values, the file holds " +
                                             std::to_string(b.shape().numel()));
                std::memcpy(out, b.data(), static_cast<std::size_t>(n) * sizeof(float));
            }
            std::shared_ptr<TiffFile> open(const std::string& path) const {
                std::lock_guard<std::mutex> g(mutex_);
                for (auto it = open_.begin(); it != open_.end(); ++it)
                    if (it->first == path) {
                        open_.splice(open_.begin(), open_, it);   // most recently used
                        return it->second;
                    }
                auto file = std::make_shared<TiffFile>(path);
                open_.emplace_front(path, file);
                while (open_.size() > kOpenFiles) open_.pop_back();
                return file;
            }

            fs::path folder_;
            DatasetMeta meta_;
            Index tiles_ = 0;
            std::vector<std::string> paths_;   // (tile, c, t) -> file, "" when the manifest has none
            mutable std::mutex mutex_;
            mutable std::list<std::pair<std::string, std::shared_ptr<TiffFile>>> open_;
        };

        struct FolderProbe {
            DatasetManifest manifest;
            DatasetMeta meta;
            std::string summary;
        };

        std::string plural(std::size_t n, const char* noun) {
            return std::to_string(n) + " " + noun + (n == 1 ? "" : "s");
        }

        // Load and validate the manifest; dims from the manifest plus one probed
        // file of the tile being opened.
        FolderProbe probeFolder(const fs::path& folder, const OpenOptions* options) {
            FolderProbe p;
            p.manifest = DatasetManifest::load(folder / DatasetManifest::kFileName);
            const std::vector<std::string> problems = p.manifest.validate(folder);
            if (!problems.empty()) {
                std::string msg = DatasetManifest::kFileName + std::string(": ") + problems.front();
                if (problems.size() > 1) msg += " (+" + std::to_string(problems.size() - 1) + " more)";
                throw std::runtime_error(msg);
            }
            const DatasetManifest& m = p.manifest;
            const Index tile = options ? options->tile : 0;
            if (tile < 0 || tile >= static_cast<Index>(m.tiles.size()))
                throw std::out_of_range("tile " + std::to_string(tile) + ": the dataset has " + plural(m.tiles.size(), "tile"));
            const ManifestFile* first = m.file(tile, 0, 0);   // validate() guarantees it
            if (!first) throw std::runtime_error("no file for the first channel and time point of tile " + m.tiles[static_cast<std::size_t>(tile)].name);
            const TiffInfo info = inspectTiff(manifestFilePath(folder, *first).string());
            if (info.pageCount() == 0) throw std::runtime_error(first->path + ": the TIFF has no pages");

            DatasetMeta& meta = p.meta;
            fs::path named = folder;
            if (named.filename().empty()) named = named.parent_path();
            meta.name = m.name.empty() ? named.filename().string() : m.name;
            meta.sourcePath = folder.string();
            meta.format = "folder";
            meta.dims = Dims5{static_cast<Index>(m.channels.size()), m.timePoints(), static_cast<Index>(info.pageCount()),
                              static_cast<Index>(info.height()), static_cast<Index>(info.width())};
            meta.sourceType = info.pixelType();
            std::error_code ec;
            for (const ManifestFile& f : m.files) {
                const std::uintmax_t size = fs::file_size(manifestFilePath(folder, f), ec);
                if (!ec) meta.bytesOnDisk += static_cast<std::uint64_t>(size);
            }
            meta.voxelUm = options && options->voxelUm ? *options->voxelUm : m.voxelUm;
            meta.frameIntervalS = m.frameIntervalS;
            meta.channels = options && options->channels ? *options->channels : m.channels;
            meta.acquisition = m.acquisition;
            meta.sim = options && options->sim ? *options->sim : m.sim;
            meta.tiles = m.tiles;
            meta.tileIndex = tile;
            meta.normalizeChannels();
            p.summary = plural(m.tiles.size(), "tile") + " · " + plural(m.channels.size(), "channel") + " · manifest";
            return p;
        }

    } // namespace

    // --- public entry points -------------------------------------------------------------

    bool zarrSupported() noexcept { return sirius::zarrSupported(); }

    std::vector<std::string> readableExtensions() {
        std::vector<std::string> out{".tif", ".tiff", ".ome.tif", ".btf"};
        if (zarrSupported()) {
            out.push_back(".zarr");
            out.push_back(".n5");
        }
        return out;
    }

    DatasetMeta probeDataset(const std::string& path) {
        std::error_code ec;
        if (!fs::exists(path, ec)) throw std::runtime_error("no such file or directory: " + path);
        if (isFolderDataset(path)) return probeFolder(path, nullptr).meta;
        if (fs::is_directory(path, ec)) {
            if (!isZarrStore(path)) throw std::runtime_error("not a zarr / N5 store: " + path);
            return probeZarr(path, nullptr).meta;
        }
        return probeTiff(path, nullptr).meta;
    }

    OpenResult openDataset(const std::string& path, const OpenOptions& options) {
        std::error_code ec;
        if (!fs::exists(path, ec)) throw std::runtime_error("no such file or directory: " + path);
        OpenResult r;
        if (isFolderDataset(path)) {
            FolderProbe p = probeFolder(path, &options);
            r.source = std::make_shared<FolderArraySource>(path, p.manifest, p.meta);
            r.metadataSummary = p.summary;
            r.dimsFromMetadata = true;
        } else if (fs::is_directory(path, ec)) {
            if (!isZarrStore(path)) throw std::runtime_error("not a zarr / N5 store: " + path);
            ZarrProbe p = probeZarr(path, &options);
            auto src = std::make_shared<ZarrArraySource>(path, p.meta, p.map);
            src->setYxLast(p.yxLast);
            r.source = src;
            r.metadataSummary = p.summary;
            r.dimsFromMetadata = true;
        } else {
            if (!looksLikeTiff(path)) {
                // let libtiff decide: many microscopy files carry odd extensions
            }
            TiffProbe p = probeTiff(path, &options);
            r.source = std::make_shared<TiffArraySource>(path, p.meta, p.order);
            r.metadataSummary = p.summary;
            r.dimsFromMetadata = p.dimsFromMetadata;
        }
        if (options.readAll) {
            DatasetMeta meta = r.source->meta();
            std::shared_ptr<Array5> all = r.source->readAll();
            r.source = std::make_shared<MemorySource>(all, meta);
        }
        r.meta = r.source->meta();
        return r;
    }

} // namespace sirius::app
