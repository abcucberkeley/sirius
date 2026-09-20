#ifndef SIRIUS_APP_OPS_LOAD_HPP
#define SIRIUS_APP_OPS_LOAD_HPP

// The Load step's reading of its own parameters, which the workbench needs
// when it opens a dataset for a pipeline that starts with one.

#include "core/array_source.hpp"
#include "core/params.hpp"

namespace sirius::app {

    // The Load step's parameters as the options it opens the dataset with
    // (page order, voxel size, SIM layout, tile, full read); 0 keeps what the
    // file says for that axis or size.
    OpenOptions loadOpenOptions(const ParamSet& loadParams);

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_LOAD_HPP
