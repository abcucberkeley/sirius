#include "sirius/otf_select.hpp"

#include "sirius/otf_io.hpp"

namespace sirius {

    OTFRadiallyAveraged selectOTF(const std::string& otfPath, const SIMParameters& p, bool threeD,
                                  const IdealOtfOptions& opts) {
        return otfPath.empty() ? idealOTF(p, threeD, opts) : loadOTF(otfPath, p);
    }

} // namespace sirius
