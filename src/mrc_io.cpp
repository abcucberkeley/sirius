#include "sirius/mrc_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "sirius/errors.hpp"

namespace sirius {

    namespace {

        namespace fs = std::filesystem;

        constexpr std::size_t kHeaderBytes = 1024;
        constexpr std::int16_t kDeltaVisionMagic = static_cast<std::int16_t>(0xC0A0);   // -16224 at byte 96
        // This computer's byte order, which decides whether a header field or a
        // sample needs its bytes reversed. The project is C++17 (cmake/
        // ProjectOptions.cmake), where there is no std::endian: the compiler's
        // own macro answers where it is defined (gcc, clang, MSVC's clang-cl)
        // and the bytes of a known word answer everywhere else. Either way the
        // answer folds to a constant.
        inline bool hostBigEndian() noexcept {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
            return __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__;
#else
            const std::uint32_t one = 1;
            unsigned char bytes[4];
            std::memcpy(bytes, &one, sizeof one);
            return bytes[0] == 0;
#endif
        }

        // The header's fields at their byte offsets, in the file's byte order.
        class Header {
        public:
            Header(const unsigned char* bytes, bool swap) : b_(bytes), swap_(swap) {}
            std::int32_t i32(std::size_t at) const {
                std::uint32_t v = 0;
                load(&v, at, 4);
                return static_cast<std::int32_t>(v);
            }
            std::int16_t i16(std::size_t at) const {
                std::uint16_t v = 0;
                load(&v, at, 2);
                return static_cast<std::int16_t>(v);
            }
            float f32(std::size_t at) const {
                std::uint32_t u = 0;
                load(&u, at, 4);
                float f = 0.f;
                std::memcpy(&f, &u, 4);
                return f;
            }

        private:
            void load(void* out, std::size_t at, std::size_t n) const {
                unsigned char tmp[8];
                std::memcpy(tmp, b_ + at, n);
                if (swap_) std::reverse(tmp, tmp + n);
                std::memcpy(out, tmp, n);
            }
            const unsigned char* b_;
            bool swap_;
        };

        bool knownMode(std::int32_t mode) noexcept {
            return mode == 0 || mode == 1 || mode == 2 || mode == 3 || mode == 4 || mode == 6 || mode == 7;
        }

        // A header that could describe a stack: positive extents a camera or a
        // microscope could produce, a pixel mode that exists, an extended
        // header that fits in a file.
        bool plausible(const Header& h) noexcept {
            const std::int32_t nx = h.i32(0), ny = h.i32(4), nz = h.i32(8), nsymbt = h.i32(92);
            return nx > 0 && nx <= (1 << 20) && ny > 0 && ny <= (1 << 20) && nz > 0 && nz <= (1 << 24) && knownMode(h.i32(12)) &&
                   nsymbt >= 0 && nsymbt <= (1 << 30);
        }

        std::uint64_t mulChecked(std::uint64_t a, std::uint64_t b, const std::string& path) {
            if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
                throw IoError(path + ": the header's extents multiply past 64 bits");
            return a * b;
        }

        std::string trimmedLabel(const unsigned char* at) {
            std::string s;
            for (std::size_t i = 0; i < 80 && at[i] != '\0'; ++i) s.push_back(static_cast<char>(at[i]));
            while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) s.pop_back();
            return s;
        }

        MrcInfo parseHeader(const unsigned char* bytes, const std::string& path, std::uint64_t bytesOnDisk) {
            // Little or big: the DeltaVision magic says, else the MRC2014 machine
            // stamp, else whichever reading is plausible (little first, as every
            // file written in the last twenty years is).
            const Header little(bytes, hostBigEndian()), big(bytes, !hostBigEndian());
            bool fileBig = false, dv = false;
            if (little.i16(96) == kDeltaVisionMagic) {
                dv = true;
            } else if (big.i16(96) == kDeltaVisionMagic) {
                dv = true;
                fileBig = true;
            } else {
                const unsigned char s0 = bytes[212], s1 = bytes[213];
                if (s0 == 0x44 && (s1 == 0x44 || s1 == 0x41)) fileBig = false;
                else if (s0 == 0x11 && s1 == 0x11) fileBig = true;
                else fileBig = !plausible(little) && plausible(big);
            }
            const Header h(bytes, fileBig != hostBigEndian());
            if (!plausible(h)) {
                const std::int32_t mode = h.i32(12);
                if (!knownMode(mode) && h.i32(0) > 0 && h.i32(4) > 0 && h.i32(8) > 0)
                    throw IoError(path + ": pixel mode " + std::to_string(mode) +
                                  " is not read (bytes, int16, float32, uint16, int32 and the complex modes 3 and 4 are)");
                throw IoError(path + ": not an MRC / DeltaVision stack (nx " + std::to_string(h.i32(0)) + ", ny " + std::to_string(h.i32(4)) +
                              ", nz " + std::to_string(h.i32(8)) + ", mode " + std::to_string(mode) + ")");
            }

            MrcInfo info;
            info.nx = static_cast<std::uint32_t>(h.i32(0));
            info.ny = static_cast<std::uint32_t>(h.i32(4));
            info.nz = static_cast<std::uint32_t>(h.i32(8));
            info.mode = h.i32(12);
            info.bigEndian = fileBig;
            info.deltaVision = dv;
            for (std::size_t k = 0; k < 3; ++k) {
                info.grid[k] = h.i32(28 + 4 * k);
                info.cell[k] = h.f32(40 + 4 * k);
            }
            info.minValue = h.f32(76);
            info.maxValue = h.f32(80);
            info.meanValue = h.f32(84);
            info.extendedHeaderBytes = static_cast<std::uint32_t>(h.i32(92));
            info.dataOffset = kHeaderBytes + info.extendedHeaderBytes;
            info.bytesOnDisk = bytesOnDisk;

            switch (info.mode) {
                case 0: info.pixelType = dv ? PixelType::UInt8 : PixelType::Int8; break;
                case 1: info.pixelType = PixelType::Int16; break;
                case 2: info.pixelType = PixelType::Float32; break;
                case 3:
                    info.pixelType = PixelType::Int16;
                    info.complex = true;
                    break;
                case 4:
                    info.pixelType = PixelType::Float32;
                    info.complex = true;
                    break;
                case 6: info.pixelType = PixelType::UInt16; break;
                case 7: info.pixelType = PixelType::Int32; break;
                default: break;   // plausible() refused the rest
            }
            info.width = info.complex ? 2 * info.nx : info.nx;
            info.height = info.ny;
            info.sections = info.nz;

            if (dv) {
                // Priism / IVE: pixel sizes in micrometres, the wavelength and
                // time axes, the sequence they interleave in.
                for (std::size_t k = 0; k < 3; ++k) info.voxelUm[k] = info.cell[k] > 0.f ? static_cast<double>(info.cell[k]) : 0.0;
                info.imageType = h.i16(160);
                info.lensId = h.i16(162);
                info.times = std::max<int>(h.i16(180), 1);
                const std::int16_t seq = h.i16(182);
                info.sequence = seq == 1 ? MrcSequence::WZT : seq == 2 ? MrcSequence::ZWT : MrcSequence::ZTW;
                info.waves = std::clamp<int>(h.i16(196), 1, 5);
                for (std::size_t k = 0; k < 5; ++k) info.wavelengthsNm[k] = h.i16(198 + 2 * k);
            } else {
                // MRC2014: Angstrom per cell, the grid counts the cells
                for (std::size_t k = 0; k < 3; ++k)
                    info.voxelUm[k] = info.grid[k] > 0 && info.cell[k] > 0.f ? static_cast<double>(info.cell[k]) / info.grid[k] * 1e-4 : 0.0;
            }
            const std::int32_t labels = std::clamp<std::int32_t>(h.i32(220), 0, 10);
            for (std::int32_t k = 0; k < labels; ++k) {
                std::string label = trimmedLabel(bytes + 224 + 80 * static_cast<std::size_t>(k));
                if (!label.empty()) info.titles.push_back(std::move(label));
            }

            const std::uint64_t sectionBytes = mulChecked(mulChecked(info.width, info.height, path), bytesPerPixel(info.pixelType), path);
            const std::uint64_t need = info.dataOffset + mulChecked(sectionBytes, info.sections, path);
            if (bytesOnDisk < need)
                throw IoError(path + ": " + std::to_string(need) + " bytes of header and sections expected, the file has " + std::to_string(bytesOnDisk));
            return info;
        }

        template <typename S, typename T>
        void convertValues(const unsigned char* raw, std::size_t n, T* dst) {
            if constexpr (std::is_same_v<S, T>) {
                std::memcpy(dst, raw, n * sizeof(T));
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    S v;
                    std::memcpy(&v, raw + i * sizeof(S), sizeof(S));
                    dst[i] = static_cast<T>(v);
                }
            }
        }

    } // namespace

    // --- MrcInfo -----------------------------------------------------------

    std::uint32_t MrcInfo::planes() const noexcept {
        const std::uint64_t per = static_cast<std::uint64_t>(std::max(waves, 1)) * static_cast<std::uint64_t>(std::max(times, 1));
        return per > 0 && sections % per == 0 ? static_cast<std::uint32_t>(sections / per) : 0;
    }

    std::uint64_t MrcInfo::sectionOf(int w, int t, std::uint32_t z) const noexcept {
        const std::uint64_t nz = planes(), nw = static_cast<std::uint64_t>(std::max(waves, 1)), nt = static_cast<std::uint64_t>(std::max(times, 1));
        const std::uint64_t wi = static_cast<std::uint64_t>(std::max(w, 0)), ti = static_cast<std::uint64_t>(std::max(t, 0));
        switch (sequence) {
            case MrcSequence::WZT: return wi + nw * (z + nz * ti);
            case MrcSequence::ZWT: return z + nz * (wi + nw * ti);
            case MrcSequence::ZTW: break;
        }
        return z + nz * (ti + nt * wi);
    }

    bool isMrcName(const std::string& path) noexcept {
        std::string ext = fs::path(path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return ext == ".dv" || ext == ".mrc";
    }

    // --- MrcFile -------------------------------------------------------------

    struct MrcFile::Impl {
        std::string path;
        MrcInfo info;
        bool swap = false;
        std::uint64_t sectionBytes = 0;
        mutable std::mutex mutex;
        mutable std::ifstream in;

        explicit Impl(std::string p) : path(std::move(p)) {
            std::error_code ec;
            const fs::path fp = fs::u8path(path);
            if (!fs::is_regular_file(fp, ec)) throw IoError(path + ": no such file");
            in.open(fp, std::ios::binary);
            if (!in) throw IoError(path + ": cannot be read");
            unsigned char bytes[kHeaderBytes];
            if (!in.read(reinterpret_cast<char*>(bytes), kHeaderBytes)) throw IoError(path + ": shorter than an MRC header (1024 bytes)");
            info = parseHeader(bytes, path, static_cast<std::uint64_t>(fs::file_size(fp, ec)));
            swap = info.bigEndian != hostBigEndian() && bytesPerPixel(info.pixelType) > 1;
            sectionBytes = static_cast<std::uint64_t>(info.width) * info.height * bytesPerPixel(info.pixelType);
        }

        // The bytes of sections [first, first + count), in the host's byte order.
        std::vector<unsigned char> raw(std::size_t first, std::size_t count) const {
            if (first > info.sections || count > info.sections - first)
                throw std::out_of_range(path + ": sections " + std::to_string(first) + ".." + std::to_string(first + count) + " of " +
                                        std::to_string(info.sections));
            std::vector<unsigned char> bytes(static_cast<std::size_t>(sectionBytes * count));
            if (bytes.empty()) return bytes;
            {
                const std::lock_guard<std::mutex> g(mutex);
                in.clear();
                in.seekg(static_cast<std::streamoff>(info.dataOffset + sectionBytes * first));
                if (!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
                    throw IoError(path + ": a read past the end of the file");
            }
            if (swap) {
                const std::size_t item = bytesPerPixel(info.pixelType);
                for (std::size_t i = 0; i + item <= bytes.size(); i += item) std::reverse(bytes.begin() + static_cast<std::ptrdiff_t>(i), bytes.begin() + static_cast<std::ptrdiff_t>(i + item));
            }
            return bytes;
        }
    };

    MrcFile::MrcFile(std::string path) : impl_(std::make_unique<Impl>(std::move(path))) {}
    MrcFile::~MrcFile() = default;
    MrcFile::MrcFile(MrcFile&&) noexcept = default;
    MrcFile& MrcFile::operator=(MrcFile&&) noexcept = default;

    const std::string& MrcFile::path() const noexcept { return impl_->path; }
    const MrcInfo& MrcFile::info() const noexcept { return impl_->info; }

    template <typename T>
    void MrcFile::readSections(std::size_t first, std::size_t count, T* dst) const {
        const std::vector<unsigned char> bytes = impl_->raw(first, count);
        const std::size_t n = static_cast<std::size_t>(impl_->info.width) * impl_->info.height * count;
        if (n == 0) return;
        switch (impl_->info.pixelType) {
            case PixelType::UInt8: convertValues<std::uint8_t>(bytes.data(), n, dst); break;
            case PixelType::Int8: convertValues<std::int8_t>(bytes.data(), n, dst); break;
            case PixelType::UInt16: convertValues<std::uint16_t>(bytes.data(), n, dst); break;
            case PixelType::Int16: convertValues<std::int16_t>(bytes.data(), n, dst); break;
            case PixelType::UInt32: convertValues<std::uint32_t>(bytes.data(), n, dst); break;
            case PixelType::Int32: convertValues<std::int32_t>(bytes.data(), n, dst); break;
            case PixelType::Float32: convertValues<float>(bytes.data(), n, dst); break;
            case PixelType::Float64: convertValues<double>(bytes.data(), n, dst); break;
        }
    }

    template <typename T>
    Buffer<T> MrcFile::readSections(std::size_t first, std::size_t count) const {
        const MrcInfo& i = impl_->info;
        Buffer<T> out(Shape{static_cast<Index>(count), static_cast<Index>(i.height), static_cast<Index>(i.width)}, Device::cpu());
        readSections<T>(first, count, out.data());
        return out;
    }

    template <typename T>
    Buffer<T> MrcFile::readStack() const {
        return readSections<T>(0, impl_->info.sections);
    }

    MrcInfo inspectMrc(const std::string& path) { return MrcFile(path).info(); }

#define SIRIUS_MRC_INSTANTIATE(T)                                                                  \
    template void MrcFile::readSections<T>(std::size_t, std::size_t, T*) const;                  \
    template Buffer<T> MrcFile::readSections<T>(std::size_t, std::size_t) const;                  \
    template Buffer<T> MrcFile::readStack<T>() const;
    SIRIUS_MRC_INSTANTIATE(std::uint8_t)
    SIRIUS_MRC_INSTANTIATE(std::int8_t)
    SIRIUS_MRC_INSTANTIATE(std::uint16_t)
    SIRIUS_MRC_INSTANTIATE(std::int16_t)
    SIRIUS_MRC_INSTANTIATE(std::uint32_t)
    SIRIUS_MRC_INSTANTIATE(std::int32_t)
    SIRIUS_MRC_INSTANTIATE(float)
    SIRIUS_MRC_INSTANTIATE(double)
#undef SIRIUS_MRC_INSTANTIATE

} // namespace sirius
