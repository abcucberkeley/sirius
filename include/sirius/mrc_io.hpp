#ifndef SIRIUS_MRC_IO_HPP
#define SIRIUS_MRC_IO_HPP

// MRC / DeltaVision stacks: the 1024-byte header of the MRC2014 standard and
// its Priism / IVE dialect that OMX microscopes write as .dv (and that
// cudasirecon reads first). Reading only; the OTF files cudasirecon's tools
// write (otf.dv) are read the same way.
//
// What the two dialects share: nx, ny, nz (sections) and the pixel mode in
// the first 16 bytes, the sampling grid and cell lengths at 28..52, the
// extended-header length (nsymbt) at 92; the sections follow at
// 1024 + nsymbt, x fastest then y, one section after another.
//
// Where they differ, and what tells them apart: a DeltaVision file carries
// the value 0xC0A0 as an int16 at byte 96, and then the cell lengths ARE the
// pixel sizes in micrometres (xlen = 0.08 for an 80 nm camera pixel), the
// number of time points is an int16 at 180, the image sequence at 182
// (0 = ZTW: z fastest, then time, then wavelength; 1 = WZT; 2 = ZWT), the
// number of wavelengths at 196 and up to five wavelengths in nanometres from
// 198. An MRC2014 file (electron microscopy: "MAP " at 208, a machine stamp
// at 212) has none of that: its cell lengths are in Angstrom per grid cell,
// so a pixel is xlen / mx * 1e-4 micrometres, and its sections are read as
// one z stack.
//
// Pixel modes: 0 bytes (unsigned in a DeltaVision file, as IVE defines them;
// signed in MRC2014), 1 int16, 2 float32, 6 uint16, 7 int32, and the two
// complex modes 3 (int16 pairs) and 4 (float32 pairs), which read as a row of
// 2 * nx values with the real and imaginary parts interleaved -- the layout
// of the otf.tif files SIRIUS already reads (sirius/otf_io.hpp). Both byte
// orders are read; the DV magic (or the MRC machine stamp) says which.
//
// Rows come back in file order, row 0 first, as cudasirecon (the IVE
// library), mrcfile and IMOD hand them out. MRC puts the origin bottom-left,
// so Bio-Formats reverses the rows when it imports a .dv: Fiji shows the
// file upside down relative to this reader, and tests/data/raw.tif, its
// export of raw.dv, is raw.dv with every row reversed -- which is why
// cudasirecon's TIFF-route config negates the k0 angles of its .dv-route
// config. Reading in file order keeps an OMX acquisition's own configuration
// valid as it is.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/pixel_type.hpp"

namespace sirius {

    // How a DeltaVision file orders its sections (header int16 at 182).
    enum class MrcSequence : std::uint8_t { ZTW = 0,   // z fastest, then time, then wavelength
                                            WZT = 1,   // wavelength fastest, then z, then time
                                            ZWT = 2 }; // z fastest, then wavelength, then time

    struct MrcInfo {
        std::uint32_t nx = 0;                  // pixels per row (complex modes: pairs per row)
        std::uint32_t ny = 0;
        std::uint32_t nz = 0;                  // sections in the file
        std::int32_t mode = 2;
        PixelType pixelType = PixelType::Float32;   // of one value a read returns
        bool complex = false;                  // mode 3 / 4: width is 2 * nx, (re, im) interleaved
        std::uint32_t width = 0;               // values per row: nx, or 2 * nx when complex
        std::uint32_t height = 0;              // == ny
        std::uint32_t sections = 0;            // == nz
        bool bigEndian = false;
        bool deltaVision = false;              // 0xC0A0 at byte 96
        std::uint32_t extendedHeaderBytes = 0; // nsymbt
        std::uint64_t dataOffset = 1024;       // 1024 + extendedHeaderBytes
        std::uint64_t bytesOnDisk = 0;
        std::array<std::int32_t, 3> grid{1, 1, 1};   // mx, my, mz
        std::array<float, 3> cell{0.f, 0.f, 0.f};    // xlen, ylen, zlen as written
        // Pixel size in micrometres, x, y, z; 0 where the file says nothing.
        // DeltaVision: the cell lengths themselves. MRC2014: cell / grid Angstrom.
        std::array<double, 3> voxelUm{0.0, 0.0, 0.0};
        float minValue = 0.f, maxValue = 0.f, meanValue = 0.f;
        // DeltaVision only (1, 1, ZTW and zeros otherwise)
        int waves = 1;
        int times = 1;
        MrcSequence sequence = MrcSequence::ZTW;
        std::array<int, 5> wavelengthsNm{0, 0, 0, 0, 0};
        int imageType = 0;
        int lensId = 0;
        std::vector<std::string> titles;       // the 80-character labels, trailing blanks removed

        // Planes per (wavelength, time point): sections / (waves * times), or 0
        // when the header's counts do not divide the sections.
        std::uint32_t planes() const noexcept;
        // The section holding plane z of wavelength w at time t, by `sequence`.
        std::uint64_t sectionOf(int w, int t, std::uint32_t z) const noexcept;
    };

    // Whether `path` is named like a file this reader opens (.dv, .mrc; any case).
    bool isMrcName(const std::string& path) noexcept;

    // Header only; throws IoError when the file is not an MRC / DeltaVision
    // stack, holds a pixel mode this reader lacks, or is shorter than its
    // header promises.
    MrcInfo inspectMrc(const std::string& path);

    // Read-only handle. Reads are serialised on one file descriptor and
    // return host buffers of shape {count, height, width}, converted from the
    // stored pixel type to T.
    class MrcFile {
    public:
        explicit MrcFile(std::string path);
        ~MrcFile();
        MrcFile(MrcFile&&) noexcept;
        MrcFile& operator=(MrcFile&&) noexcept;
        MrcFile(const MrcFile&) = delete;
        MrcFile& operator=(const MrcFile&) = delete;

        const std::string& path() const noexcept;
        const MrcInfo& info() const noexcept;

        // Sections [first, first + count) into `dst` (count * height * width
        // values, host memory). std::out_of_range past the last section.
        template <typename T>
        void readSections(std::size_t first, std::size_t count, T* dst) const;

        template <typename T>
        Buffer<T> readSections(std::size_t first, std::size_t count) const;

        // Every section: {sections, height, width}.
        template <typename T>
        Buffer<T> readStack() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius

#endif // SIRIUS_MRC_IO_HPP
