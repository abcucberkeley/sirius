#ifndef SIRIUS_OTF_SELECT_HPP
#define SIRIUS_OTF_SELECT_HPP

// Which OTF a reconstruction uses: the file when one is named, the theoretical
// one when none is.
//
// One function, in the library, because every front has to make the same
// choice. Until 2026-10-08 the choice was a ternary inside
// app/core/session.cpp, which the Python bindings do not link: `step_sim`
// therefore refused to run at all without a measured OTF file, and since the
// application's own `export_python` writes a script that calls
// `run_pipeline`, EVERY default-OTF SIM step exported from the GUI or the CLI
// died on importing its own pipeline (docs/findings.md 9k.50, finding 4).

#include "sirius/otf.hpp"
#include "sirius/otf_ideal.hpp"
#include "sirius/sim_parameters.hpp"

#include <string>

namespace sirius {

    // `otfPath` empty: the theoretical OTF, idealOTF(p, threeD, opts) -- which
    // carries its own radial and axial steps, so it needs nothing from the
    // stack but its dimensionality. Otherwise the radially averaged OTF in
    // that file, loadOTF(otfPath, p), whose radial step is derived from p.dx.
    //
    // `threeD` selects the 3D theoretical OTF (missing cone, order 1 shifted
    // by the illumination's kz) over the in-focus 2D one, and is ignored for a
    // file. A raw stack decides it: SIMParameters::planes(sections) > 1.
    OTFRadiallyAveraged selectOTF(const std::string& otfPath, const SIMParameters& p, bool threeD,
                                  const IdealOtfOptions& opts = {});

} // namespace sirius

#endif // SIRIUS_OTF_SELECT_HPP
