#include "core/app_paths.hpp"

#include <filesystem>
#include <mutex>
#include <system_error>

namespace sirius::app {

    namespace {
        namespace fs = std::filesystem;

        std::mutex& pathMutex() {
            static std::mutex m;
            return m;
        }

        std::string& applicationDir() {
            static std::string dir;
            return dir;
        }

        std::string existingDirectory(const fs::path& dir) {
            std::error_code ec;
            if (!fs::is_directory(dir, ec)) return {};
            return dir.lexically_normal().string();
        }
    } // namespace

    void setApplicationDirectory(const std::string& dir) {
        std::lock_guard<std::mutex> g(pathMutex());
        applicationDir() = dir;
    }

    std::string applicationDirectory() {
        std::lock_guard<std::mutex> g(pathMutex());
        return applicationDir();
    }

    std::string installedDataDirectoryFromBindir() {
#ifdef SIRIUS_APP_DATADIR_FROM_BINDIR
        return SIRIUS_APP_DATADIR_FROM_BINDIR;
#else
        return {};
#endif
    }

    std::string installedDataDirectory(const std::string& name) {
        const std::string app = applicationDirectory();
        const std::string relative = installedDataDirectoryFromBindir();
        if (app.empty() || relative.empty() || name.empty()) return {};
        return existingDirectory(fs::path(app) / relative / name);
    }

    std::string besideApplication(const std::string& name) {
        const std::string app = applicationDirectory();
        if (app.empty() || name.empty()) return {};
        return existingDirectory(fs::path(app) / name);
    }

} // namespace sirius::app
