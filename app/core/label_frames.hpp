#ifndef SIRIUS_APP_LABEL_FRAMES_HPP
#define SIRIUS_APP_LABEL_FRAMES_HPP

// What the label volume (core/labels.hpp) and the track index
// (core/tracks.hpp) both speak: the voxels of a labelled clip as a plain view,
// the diff of one edit, and the lineage map. On its own so that the index can
// sit *below* the volume that owns one -- it reads frames and diffs, and has
// no use for the rest of LabelVolume -- instead of the two headers including
// each other.

#include <cstdint>
#include <map>
#include <vector>

#include <sirius/index.hpp>

namespace sirius::app {

    // {child track id: parent track id}; divisions only.
    using Lineage = std::map<std::uint32_t, std::uint32_t>;

    // Voxel diff of one edit: linear indices into one (z, y, x) volume of
    // time point t, with the values before and after.
    struct LabelDiff {
        Index t = 0;
        std::vector<Index> indices;
        std::vector<std::uint32_t> before;
        std::vector<std::uint32_t> after;

        bool empty() const noexcept { return indices.empty(); }
    };

    // The voxels of a label volume, read-only and not owned: t frames of
    // (z, y, x) ids, contiguous, 0 = background (LabelVolume::frames()). Valid
    // as long as the volume it came from is neither written to nor destroyed.
    struct LabelFrames {
        const std::uint32_t* data = nullptr;
        Index t = 0, z = 0, y = 0, x = 0;

        Index volumeSize() const noexcept { return z * y * x; }
        bool empty() const noexcept { return !data || t <= 0 || volumeSize() <= 0; }
        const std::uint32_t* frame(Index i) const noexcept { return data + i * volumeSize(); }
    };

} // namespace sirius::app

#endif // SIRIUS_APP_LABEL_FRAMES_HPP
