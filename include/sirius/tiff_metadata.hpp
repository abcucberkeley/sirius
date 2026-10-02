#ifndef SIRIUS_TIFF_METADATA_HPP
#define SIRIUS_TIFF_METADATA_HPP

// What a microscopy TIFF says about itself in its first ImageDescription:
// OME-XML (OME-TIFF) or ImageJ's "key=value" header (ImageJ hyperstacks).
// Parsed here, in the library, so that neither the workbench nor the Python
// worker needs a second reader for it. The parser is deliberately lenient:
// microscopy software writes every variant of both, and an attribute that
// does not parse reads as "not given" (0 / empty), never as an error.

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace sirius {

    // One OME <Channel>.
    struct OmeChannel {
        std::string id;
        std::string name;
        std::uint32_t samplesPerPixel = 1;   // 3 for an RGB channel stored as one sample-interleaved plane
        double emissionNm = 0.0;             // 0 = not given
        double excitationNm = 0.0;
        bool hasColor = false;
        std::uint32_t colorRgba = 0xFFFFFFFFu;   // OME Color: r << 24 | g << 16 | b << 8 | a
        std::string fluor;
    };

    // One OME <TiffData> block: which IFDs hold which planes.
    struct OmeTiffData {
        std::uint32_t ifd = 0;          // first IFD (main-chain page index) of the block
        std::uint32_t firstC = 0, firstZ = 0, firstT = 0;
        std::uint32_t planeCount = 0;   // 0 = not given (1 when IFD is given, else every plane)
        bool hasIfd = false;
        bool hasPlaneCount = false;
        std::string uuid;               // <UUID>: the file holding the block; "" = this file
        std::string fileName;           // <UUID FileName=...>
    };

    // One OME <Image> (a "series" in Bio-Formats / tifffile terms).
    struct OmeImage {
        std::string id;
        std::string name;
        std::string dimensionOrder;   // "XYCZT", ...
        std::string type;             // "uint16", "float", ...
        std::uint64_t sizeX = 0, sizeY = 0, sizeZ = 0, sizeC = 0, sizeT = 0;
        std::array<double, 3> physicalSizeUm{0.0, 0.0, 0.0};   // x, y, z; 0 = not given
        double timeIncrementS = 0.0;
        bool interleaved = false;
        std::vector<OmeChannel> channels;
        std::vector<OmeTiffData> tiffData;
    };

    // ImageJ's description: "ImageJ=1.54f\nimages=24\nchannels=2\n...".
    struct ImageJMetadata {
        std::string version;
        std::uint32_t images = 0, channels = 0, slices = 0, frames = 0;   // 0 = not given
        bool hyperstack = false;
        std::string mode;          // "composite", "color", "grayscale"
        std::string unit;          // as written ("micron", "\\u00B5m", "nm", ...)
        double unitUm = 0.0;       // the unit in micrometres; 0 = not a length (or "pixel")
        double spacing = 0.0;      // z step, in `unit`
        double frameInterval = 0.0;   // finterval, seconds
        double min = 0.0, max = 0.0;  // display range
        bool hasRange = false;
        std::map<std::string, std::string> entries;   // every key=value line
    };

    // Channel name / wavelength / colour, whichever format said it.
    struct TiffChannel {
        std::string name;
        double emissionNm = 0.0;
        bool hasColor = false;
        std::array<float, 3> color{1.f, 1.f, 1.f};   // linear 0..1
    };

    struct TiffMetadata {
        bool ome = false;
        bool imagej = false;
        std::vector<OmeImage> omeImages;   // OME: every <Image>, in document order
        std::string omeUuid;               // OME: the root element's UUID (this file's)
        ImageJMetadata imageJ;             // ImageJ only

        // The first image (series) in one vocabulary, whichever format: 0 /
        // empty where the file does not say.
        std::uint64_t sizeC = 0, sizeZ = 0, sizeT = 0;
        std::string dimensionOrder;              // OME DimensionOrder; "XYCZT" for ImageJ hyperstacks
        std::array<double, 3> voxelUm{0.0, 0.0, 0.0};   // x, y, z; ImageJ: z only (x / y come from the resolution tags)
        double frameIntervalS = 0.0;
        std::vector<TiffChannel> channels;       // OME channels of the first image
    };

    // Parse an ImageDescription; returns ome == imagej == false for anything else.
    TiffMetadata parseTiffMetadata(const std::string& description);

    // A length unit name ("um", "µm", "micron", "nm", "mm", "cm", "m", "inch")
    // in micrometres; "" means micrometres; "pixel" is 0; anything unknown 1.
    double tiffUnitToUm(const std::string& unit);

    // Page indices (into TiffInfo::pages, i.e. main-chain full-resolution
    // IFDs) of each OME image, in the image's DimensionOrder plane order.
    // Images without TiffData blocks take the pages after the previous one.
    // Planes in another file (a TiffData UUID that is not this file's) or
    // beyond `pageCount` end that image's list early: it is then shorter than
    // SizeC / samplesPerPixel * SizeZ * SizeT.
    std::vector<std::vector<std::uint32_t>> omeImagePages(const TiffMetadata& md, std::size_t pageCount,
                                                          std::uint32_t samplesPerPixel = 1);

} // namespace sirius

#endif // SIRIUS_TIFF_METADATA_HPP
