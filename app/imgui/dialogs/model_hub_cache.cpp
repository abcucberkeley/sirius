#include "imgui/dialogs/model_hub_cache.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <system_error>

#include "imgui/platform.hpp"
#include "imgui/strings.hpp"

namespace sirius::app::gui::modelhub {

    namespace {

        namespace fs = std::filesystem;

        fs::path fromUtf8(const std::string& s) { return fs::u8path(s); }
        // In the platform's separators, as the worker's str(Path) writes them:
        // a spec chosen here and one the worker reports compare equal.
        std::string toUtf8(fs::path p) { return p.make_preferred().u8string(); }

        // The path with its links and dots resolved, whether or not it exists.
        fs::path resolved(const fs::path& p) {
            std::error_code ec;
            fs::path out = fs::weakly_canonical(p, ec);
            if (ec) out = fs::absolute(p, ec).lexically_normal();
            return out;
        }

        // `p` is `root` or lies below it (both resolved).
        bool isWithin(const fs::path& p, const fs::path& root) {
            const fs::path rel = p.lexically_relative(root);
            if (rel.empty()) return false;
            return *rel.begin() != fs::path("..");
        }

        bool isRegularFile(const fs::path& p) {
            std::error_code ec;
            return fs::is_regular_file(p, ec);
        }

        std::uint64_t sizeOf(const fs::path& p) {
            std::error_code ec;
            const std::uintmax_t n = fs::file_size(p, ec);
            return ec ? 0 : static_cast<std::uint64_t>(n);
        }

        // The entries of a directory sorted by name, as the worker's sorted() lists them.
        std::vector<fs::path> sortedEntries(const fs::path& dir, bool recursive) {
            std::vector<fs::path> out;
            std::error_code ec;
            if (recursive) {
                for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
                     it.increment(ec))
                    out.push_back(it->path());
            } else {
                for (fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
                     it.increment(ec))
                    out.push_back(it->path());
            }
            std::sort(out.begin(), out.end());
            return out;
        }

        fs::path cacheRoot() {
            const std::string env = trimmed(platform::environment("SIRIUS_MODEL_CACHE"));
            if (env.empty()) return fromUtf8(platform::homeDirectory()) / ".sirius" / "models";
            if (env == "~") return fromUtf8(platform::homeDirectory());
            if (startsWith(env, "~/") || startsWith(env, "~\\")) return fromUtf8(platform::homeDirectory()) / fromUtf8(env.substr(2));
            return fromUtf8(env);
        }

        fs::path repositoryRoot(const std::string& repo) {
            // one flat directory per repository; "/" is not a legal file-name character
            return cacheRoot() / "hf" / fromUtf8(replaceAll(repo, "/", "--"));
        }

    } // namespace

    bool isModelFile(const std::string& name) {
        for (const char* ext : {".pt", ".pts", ".pth", ".onnx"})
            if (endsWithNoCase(name, ext)) return true;
        return false;
    }

    std::string cacheDirectory() { return toUtf8(cacheRoot()); }

    std::string repositoryDirectory(const std::string& repo) { return toUtf8(repositoryRoot(repo)); }

    std::string downloadTarget(const std::string& repo, const std::string& file) {
        const fs::path root = resolved(repositoryRoot(repo));
        const fs::path target = resolved(repositoryRoot(repo) / fromUtf8(file));
        if (file.empty() || target == root || !isWithin(target, root))
            throw std::runtime_error(repo + ": '" + file + "' is not a file name inside the repository");
        return toUtf8(target);
    }

    std::string cachedPath(const std::string& repo, const std::string& file) {
        if (file.empty()) return std::string();
        try {
            const std::string target = downloadTarget(repo, file);
            return isRegularFile(fromUtf8(target)) ? target : std::string();
        } catch (const std::exception&) {
            return std::string();
        }
    }

    std::vector<CachedModel> listCachedModels() {
        std::vector<CachedModel> out;
        const fs::path root = cacheRoot();
        std::error_code ec;
        const fs::path hf = root / "hf";
        if (fs::is_directory(hf, ec)) {
            for (const fs::path& dir : sortedEntries(hf, false)) {
                if (!fs::is_directory(dir, ec)) continue;
                // the first "--" is the one that stood for the "/" of owner/repo
                std::string repo = toUtf8(dir.filename());
                const std::size_t dashes = repo.find("--");
                if (dashes != std::string::npos) repo.replace(dashes, 2, "/");
                for (const fs::path& f : sortedEntries(dir, true)) {
                    const std::string name = toUtf8(f.filename());
                    if (!isRegularFile(f) || !isModelFile(name) || startsWith(name, ".")) continue;
                    CachedModel m;
                    m.file = replaceAll(toUtf8(f.lexically_relative(dir)), "\\", "/");
                    m.repo = repo;
                    m.spec = "hf:" + repo + ":" + m.file;
                    m.path = toUtf8(f);
                    m.bytes = sizeOf(f);
                    out.push_back(std::move(m));
                }
            }
        }
        const fs::path local = root / "local";
        if (fs::is_directory(local, ec)) {
            for (const fs::path& f : sortedEntries(local, false)) {
                const std::string name = toUtf8(f.filename());
                if (!isRegularFile(f) || !isModelFile(name)) continue;
                CachedModel m;
                m.file = name;
                m.spec = m.path = toUtf8(f);
                m.bytes = sizeOf(f);
                out.push_back(std::move(m));
            }
        }
        return out;
    }

    Deleted deleteCachedModel(const std::string& path) {
        if (path.empty()) throw std::runtime_error("no model given to delete");
        const fs::path root = resolved(cacheRoot());
        const fs::path target = resolved(fromUtf8(path));
        if (!isWithin(target, root))
            throw std::runtime_error(path + " is not in the model cache (" + toUtf8(root) + "); delete it yourself if you meant to");
        if (target == root) {
            // The root is "within" itself, and naming the cache directory
            // would otherwise remove every cached model.
            throw std::runtime_error(path + " is the model cache itself, not a model in it");
        }
        std::error_code ec;
        if (!fs::exists(target, ec)) throw std::runtime_error(path + " is already gone");

        Deleted out;
        out.path = toUtf8(target);
        if (fs::is_directory(target, ec)) {
            for (const fs::path& f : sortedEntries(target, true))
                if (isRegularFile(f)) out.bytes += sizeOf(f);
            ec.clear();
            fs::remove_all(target, ec);
        } else {
            out.bytes = sizeOf(target);
            ec.clear();
            fs::remove(target, ec);
        }
        if (ec) throw std::runtime_error("cannot delete " + path + ": " + ec.message());

        // prune the directories the file left behind, but never the cache itself
        fs::path parent = target.parent_path();
        while (parent != root && isWithin(parent, root)) {
            ec.clear();
            if (!fs::is_empty(parent, ec) || ec) break;
            if (!fs::remove(parent, ec) || ec) break;
            out.removedDirectories.push_back(toUtf8(parent));
            parent = parent.parent_path();
        }
        return out;
    }

} // namespace sirius::app::gui::modelhub
