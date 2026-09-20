#ifndef SIRIUS_INDEX_HPP
#define SIRIUS_INDEX_HPP

// The integer every extent, offset and element count is written in. It is
// Eigen's index type (buffer.hpp asserts that), spelled here without Eigen so
// that code which only walks raw pointers -- the image operations, the box
// down-sampler -- does not have to include the tensor library, or link the
// buffer layer, to name it.

#include <cstddef>

namespace sirius {

    using Index = std::ptrdiff_t;

} // namespace sirius

#endif // SIRIUS_INDEX_HPP
