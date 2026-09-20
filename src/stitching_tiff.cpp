#include "sirius/stitching_tiff.hpp"

#include <cstdint>
#include <stdexcept>

namespace sirius {

    template <typename T>
    Buffer<T> stitchTiffTiles(const std::vector<StitchTile>& tiles, const StitchOptions& options,
                              StitchLayout* layout, const std::string& outputPath,
                              TiffCompression compression) {
        if (tiles.empty()) throw std::invalid_argument("stitchTiffTiles: no tiles given");

        std::vector<Buffer<T>> data;
        std::vector<BufferView<const T>> views;
        std::vector<std::array<double, 3>> nominal;
        data.reserve(tiles.size());
        nominal.reserve(tiles.size());
        for (const StitchTile& t : tiles) {
            data.push_back(TiffFile(t.path).readStack<T>());
            nominal.push_back(t.position);
        }
        views.reserve(data.size());
        for (const Buffer<T>& d : data) views.push_back(d.view());

        const StitchLayout plan = planStitch<T>(views, nominal, options);
        Buffer<T> fused = fuseTiles<T>(views, plan.positions, plan.canvasOrigin, plan.canvasExtent,
                                       options);
        if (!outputPath.empty()) writeTiffStack<T>(outputPath, fused.view(), compression);
        if (layout) *layout = plan;
        return fused;
    }

#define SIRIUS_INSTANTIATE_STITCH_TIFF(T)                                                       \
    template Buffer<T> stitchTiffTiles<T>(const std::vector<StitchTile>&, const StitchOptions&, \
                                          StitchLayout*, const std::string&, TiffCompression);

    SIRIUS_INSTANTIATE_STITCH_TIFF(double)
    SIRIUS_INSTANTIATE_STITCH_TIFF(float)
    SIRIUS_INSTANTIATE_STITCH_TIFF(std::uint16_t)
    SIRIUS_INSTANTIATE_STITCH_TIFF(std::uint8_t)
#undef SIRIUS_INSTANTIATE_STITCH_TIFF

} // namespace sirius
