#ifndef SIRIUS_APP_APP_PATHS_HPP
#define SIRIUS_APP_APP_PATHS_HPP

// Where the directories that ship with the application are -- the help pages,
// the Python worker, the example plugins -- in the three layouts a binary runs
// from:
//
//   installed   <prefix>/bin/sirius-app beside <prefix>/share/sirius/<name>
//               (cmake --install; the relative path is the install's
//               datadir as seen from its bindir, fixed at configure time)
//   build tree  <build>/app/sirius-app, the directories copied beside it
//   checkout    the source tree the binary was built from (SIRIUS_APP_SOURCE_DIR)
//
// Each caller keeps its own order between them; this only answers where a
// layout would put a directory and whether it is there.

#include <string>

namespace sirius::app {

    // The directory of the running executable. main() sets it; it stays
    // empty where nothing does (the tests), and then neither lookup below
    // finds anything.
    void setApplicationDirectory(const std::string& dir);
    std::string applicationDirectory();

    // The installed data directory as seen from the installed executable's
    // directory ("../share/sirius" for the default GNUInstallDirs); "" in a
    // build configured without one.
    std::string installedDataDirectoryFromBindir();
    // <exe dir>/<datadir from bindir>/<name> when that directory exists, else "".
    std::string installedDataDirectory(const std::string& name);
    // <exe dir>/<name> when that directory exists, else "".
    std::string besideApplication(const std::string& name);

} // namespace sirius::app

#endif // SIRIUS_APP_APP_PATHS_HPP
