#ifndef SIRIUS_APP_DATASET_HPP
#define SIRIUS_APP_DATASET_HPP

// Metadata that travels with every array through the pipeline: what the axes
// mean physically (voxel size, frame interval), how the channels are named
// and coloured, where the data came from and, for raw SIM acquisitions, how
// the file axes hold the (angle, phase, z) frames.

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <sirius/pixel_type.hpp>

#include "core/array.hpp"

namespace sirius::app {

    struct ChannelInfo {
        std::string label;                      // "α-actinin"
        double wavelengthNm = 0.0;              // emission; 0 = unknown
        std::array<float, 3> color{1.f, 1.f, 1.f};   // display colour, linear 0..1
        std::string exposure;                   // free text, e.g. "8 ms"

        // "488" or the label when there is no wavelength.
        std::string shortName() const;
        std::string hexColor() const;           // "#63e08a"
    };

    // Default display colour for an emission wavelength (the design's channel
    // palette: 405 blue, 488 green, 561 magenta, 640 orange; others interpolated).
    std::array<float, 3> colorForWavelength(double nm) noexcept;
    std::array<float, 3> colorFromHex(const std::string& hex);

    // How a raw structured-illumination acquisition is stored: which file
    // axes hold the logical SIM axes angle, phase and z. The shorthand that
    // every pipeline has used -- ndirs, nphases, fastSi -- says the z axis
    // holds ndirs * nphases * nz sections in angle -> z -> phase order (z ->
    // angle -> phase when fastSi). The general form, `storage`, is a text
    // (parseSimStorage) that can also put the angles on the channel axis or
    // tile angle x phase inside each plane; when it is set it is the layout
    // and ndirs / nphases mirror the angle and phase counts it names, so
    // every reader of the two keeps working. A storage layout is an
    // illumination-free description: nothing here says what the pattern is.
    struct SimLayout {
        bool present = false;
        int ndirs = 3;
        int nphases = 5;
        bool fastSi = false;
        std::string storage;   // "" = the shorthand; else SimStorage::text()

        Index sectionsPerPlane() const noexcept { return static_cast<Index>(ndirs) * nphases; }
        bool isShorthand() const noexcept { return storage.empty(); }
        // The layout as text: `storage`, or the shorthand expanded
        // ("z=[angle 3, z, phase 5]"; "z=[z, angle 3, phase 5]" when fastSi).
        std::string text() const;
        // A present layout from its text (throws std::invalid_argument as
        // parseSimStorage does); ndirs / nphases / fastSi follow the text.
        static SimLayout fromText(const std::string& text);
        // The shorthand: ndirs angles x nphases phases on z.
        static SimLayout shorthand(int ndirs, int nphases, bool fastSi = false);

        friend bool operator==(const SimLayout& a, const SimLayout& b) noexcept {
            return a.present == b.present && a.ndirs == b.ndirs && a.nphases == b.nphases && a.fastSi == b.fastSi && a.storage == b.storage;
        }
        friend bool operator!=(const SimLayout& a, const SimLayout& b) noexcept { return !(a == b); }
    };

    // One logical axis a component of a file axis holds.
    enum class SimAxis { Angle,
                         Phase,
                         Z,
                         C,    // the real channels (only on the c axis)
                         T };  // the real time points (only on the t axis)
    const char* simAxisName(SimAxis a) noexcept;   // "angle" "phase" "z" "c" "t"

    // One component of a file axis: a logical axis and how many of it.
    // extent 0 is the remainder -- what the file axis holds once the named
    // extents are divided out -- allowed only for the axis's own kind (c on
    // c, t on t, z on z), because angle and phase counts are a statement
    // about the acquisition and are always written down.
    struct SimFactor {
        SimAxis axis = SimAxis::Z;
        Index extent = 0;
        friend bool operator==(const SimFactor& a, const SimFactor& b) noexcept { return a.axis == b.axis && a.extent == b.extent; }
    };

    // The storage layout parsed. Three mechanisms:
    //   identity        c=angle 3                 the c axis is the angle axis (3 of them)
    //   factorisation   z=[angle 3, z, phase 5]   one axis multiplexes several, outermost first;
    //                                             z=[z, angle 3, phase 5] is the fast-SI order
    //   montage         yx=3x3[angle 3, phase 3]  a rows x cols grid of tiles inside every plane,
    //                                             tiles numbered row by row, the factors nested
    //                                             across that number outermost first
    // Entries are separated by ';'. A file axis without an entry is itself
    // (c stays channels, t time points, z planes). Every logical axis
    // appears at most once; a layout names angle or phase at least once.
    // yx=[angle 3, phase 3] without a grid is 3 rows (the first factor) by
    // the rest.
    struct SimStorage {
        std::array<std::vector<SimFactor>, 3> axes;   // the c, t, z axes; empty = identity
        Index rows = 1, cols = 1;                     // the montage grid; 1 x 1 = none
        std::vector<SimFactor> tiles;                 // across the tile number, outermost first
        bool montage() const noexcept { return !tiles.empty(); }
        // Canonical text: entries c, t, z, yx in that order, one space after
        // commas and semicolons, no brackets around a single factor.
        std::string text() const;
        // The angle / phase extent it names (1 when it names none).
        Index angles() const noexcept;
        Index phases() const noexcept;
    };
    // Throws std::invalid_argument naming what is wrong with the text.
    SimStorage parseSimStorage(const std::string& text);
    // The storage of a layout: parsed from `storage`, or the shorthand expanded.
    SimStorage simStorageOf(const SimLayout& layout);

    // A storage layout bound to the dims of a dataset: every extent known
    // and the arithmetic between a logical frame and where the file keeps
    // it. bindSimLayout checks the extents -- the product assigned to an
    // axis must be its length, or divide it when the axis's own kind takes
    // the remainder -- and throws std::invalid_argument quoting the numbers
    // ("z holds 134 sections, not a multiple of angle 3 × phase 5 = 15.").
    struct SimFrames {
        Index angles = 1, phases = 1, nz = 1;   // the logical extents
        Index channels = 1, times = 1;          // real channels and time points left over
        Index rows = 1, cols = 1;               // the montage grid
        Index tileY = 0, tileX = 0;             // one tile's rows and columns (the plane's without a montage)
        Dims5 dims;                             // what it was bound to
        SimStorage storage;                     // with the remainders filled in

        // Where a logical frame is: the (c, t, z) plane of the file and the tile in it.
        struct Frame {
            Index c = 0, t = 0, z = 0, row = 0, col = 0;
            friend bool operator==(const Frame& a, const Frame& b) noexcept {
                return a.c == b.c && a.t == b.t && a.z == b.z && a.row == b.row && a.col == b.col;
            }
        };
        // A logical frame: the SIM coordinates and the real channel / time point.
        struct Logical {
            Index angle = 0, phase = 0, z = 0, c = 0, t = 0;
            friend bool operator==(const Logical& a, const Logical& b) noexcept {
                return a.angle == b.angle && a.phase == b.phase && a.z == b.z && a.c == b.c && a.t == b.t;
            }
        };
        Frame frameOf(const Logical& l) const;      // throws std::out_of_range off the layout
        Logical logicalOf(const Frame& f) const;    // the inverse
        // Section index on the z axis of (angle, phase, z) for real channel 0,
        // time point 0 -- the number a z-packed stack is indexed by. For the
        // shorthand this is (angle * nz + z) * nphases + phase, or
        // (z * ndirs + angle) * nphases + phase when fastSi.
        Index sectionIndex(Index angle, Index phase, Index z) const { return frameOf({angle, phase, z, 0, 0}).z; }
        Index sections() const noexcept { return angles * phases * nz; }
        // Whether a (c, t) volume read as it is stored is already the stack
        // the reconstructor takes: everything on z, nothing on c, t or in
        // the plane, in one of the two orders it knows. The value is the
        // order (true: fast SI); nullopt means the frames have to be gathered.
        std::optional<bool> libraryOrder() const;
    };
    SimFrames bindSimLayout(const SimLayout& layout, const Dims5& dims);
    // "" when the layout binds to `dims`; else what bindSimLayout would throw.
    std::string simLayoutProblem(const SimLayout& layout, const Dims5& dims);

    // One tile of a multi-file dataset (a folder described by a manifest):
    // every tile has the same (c, t, z, y, x) shape and a nominal origin.
    struct TileInfo {
        std::string name;                       // "tile_1_2"
        std::array<double, 3> positionUm{0, 0, 0};   // nominal origin, z, y, x in micrometres
        std::array<Index, 3> gridIndex{0, 0, 0};     // z, row, col when the tiles form a grid
    };

    struct DatasetMeta {
        std::string name;                       // display name (file stem)
        std::string sourcePath;                 // file / directory the Load step reads
        std::string format;                     // "tiff", "ome-tiff", "zarr", "n5", "memory"
        Dims5 dims;
        PixelType sourceType = PixelType::Float32;   // dtype on disk
        std::uint64_t bytesOnDisk = 0;
        std::array<double, 3> voxelUm{0.1, 0.1, 0.2};   // x, y, z
        double frameIntervalS = 0.0;            // 0 = unknown
        std::vector<ChannelInfo> channels;      // size == dims.c (or 3 when rgb)
        std::string acquisition;                // "3D-SIM raw · 15 phases", "Lattice light-sheet"
        SimLayout sim;
        bool rgb = false;                       // the c axis holds display R, G, B
        bool lightSheet = false;                // acquired at an angle: deskew applies
        double sheetAngleDeg = 0.0;
        // Multi-file datasets: the tiles the folder holds and the one the
        // array (dims) describes; every tile has the same dims.
        std::vector<TileInfo> tiles;
        Index tileIndex = 0;
        bool hasTiles() const noexcept { return tiles.size() > 1; }
        // Nominal tile origins in voxels of this dataset (from positionUm / voxelUm).
        std::vector<std::array<double, 3>> tilePositionsPx() const;

        double dx() const noexcept { return voxelUm[0]; }
        double dy() const noexcept { return voxelUm[1]; }
        double dz() const noexcept { return voxelUm[2]; }
        // Channel list resized to the array's c, colours assigned by wavelength.
        void normalizeChannels();
        // "c2 t40 z48 y2048 x2048", "rgb z48 …" when rgb
        std::string shapeString() const;
        // "0.032 × 0.032 × 0.110 µm"
        std::string voxelString() const;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_DATASET_HPP
