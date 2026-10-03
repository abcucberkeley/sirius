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

    // Whether the Load step's Source names a folder dataset rather than a
    // file: the Source field's File | Folder switch shows it, on this
    // computer and on the cluster alike. A folder of TIFF stacks (with or
    // without its sirius-dataset.toml), a manifest .toml, a zarr / N5 store.
    // A path on this computer is looked at; a cluster path
    // (cluster://host/...) is judged by its name: a trailing slash, .zarr,
    // .n5 or .toml, or a last component without an extension. "" is a file.
    bool loadSourceIsFolder(const std::string& path);

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_LOAD_HPP
