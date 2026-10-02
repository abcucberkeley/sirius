#include "core/host.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <wchar.h>
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <signal.h>
#include <grp.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <cstdint>
#include <mach-o/dyld.h>
#include <spawn.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
extern char** environ;
#endif

namespace sirius::app::host {

    namespace {
        namespace fs = std::filesystem;

        // Empty for bytes that are not UTF-8: u8path throws on them on
        // Windows, and nothing here may throw.
        fs::path fsPath(const std::string& p) {
            try {
                return fs::u8path(p);
            } catch (const std::exception&) {
                return {};
            }
        }

#ifdef _WIN32
        std::wstring widen(const std::string& s) {
            if (s.empty()) return std::wstring();
            const int n = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
            std::wstring w(static_cast<std::size_t>(n), L'\0');
            ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
            return w;
        }
        std::string narrow(const std::wstring& w) {
            if (w.empty()) return std::string();
            const int n = ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
            std::string s(static_cast<std::size_t>(n), '\0');
            ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
            return s;
        }
        std::string wideEnvironment(const wchar_t* name) {
            const DWORD n = ::GetEnvironmentVariableW(name, nullptr, 0);
            if (n == 0) return std::string();
            std::wstring w(n, L'\0');
            const DWORD got = ::GetEnvironmentVariableW(name, w.data(), n);
            w.resize(got);
            return narrow(w);
        }
#endif

        // A path as UTF-8. path::u8string() throws on Windows for a name that
        // is no valid UTF-16 (an unpaired surrogate, which NTFS allows); such
        // a character becomes U+FFFD here instead.
        std::string utf8(const fs::path& p) {
#ifdef _WIN32
            return narrow(p.native());
#else
            return p.native();
#endif
        }

        // Windows paths as the rest of the application writes them. Elsewhere
        // a backslash is an ordinary character of a file name, left alone.
        std::string forwardSlashes(std::string s) {
#ifdef _WIN32
            for (char& c : s)
                if (c == '\\') c = '/';
#endif
            return s;
        }

        // Copies of the GUI's string helpers (imgui/strings), which core
        // code cannot include.
        std::vector<std::string> split(std::string_view s, char sep, bool skipEmpty = false) {
            std::vector<std::string> out;
            std::size_t start = 0;
            while (start <= s.size()) {
                const std::size_t end = s.find(sep, start);
                const std::string_view part = s.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start);
                if (!part.empty() || !skipEmpty) out.emplace_back(part);
                if (end == std::string_view::npos) break;
                start = end + 1;
            }
            return out;
        }

#ifdef _WIN32
        std::string toLower(std::string_view s) {
            std::string out(s);
            for (char& c : out)
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
            return out;
        }

        bool startsWith(std::string_view s, std::string_view prefix) {
            return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
        }

        bool endsWith(std::string_view s, std::string_view suffix) {
            return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
        }

        // The minor version `name` spells right after `prefix` ("python3." 13
        // ".exe", "python3" 13 "-arm64"), as a number, with `rest` set to what
        // follows its digits; -1 when no digits follow the prefix.
        int minorVersion(std::string_view name, std::string_view prefix, std::string_view& rest) {
            if (!startsWith(name, prefix)) return -1;
            std::size_t n = prefix.size();
            int minor = 0;
            while (n < name.size() && n - prefix.size() < 3 && name[n] >= '0' && name[n] <= '9')
                minor = minor * 10 + (name[n++] - '0');
            if (n == prefix.size()) return -1;
            rest = name.substr(n);
            return minor;
        }

        // %LOCALAPPDATA%\Microsoft\WindowsApps holds the Store's app
        // execution aliases: python.exe there is a zero-byte reparse point
        // that prints an advertisement when Python is not installed.
        bool isStoreAlias(const fs::path& p) { return toLower(utf8(p)).find("windowsapps") != std::string::npos; }

        // A sharing violation or a denied access that another process causes
        // for a moment: a virus scanner, a backup or sync tool, an indexer.
        bool transient(const std::error_code& ec) {
            return ec.value() == ERROR_ACCESS_DENIED || ec.value() == ERROR_SHARING_VIOLATION || ec.value() == ERROR_LOCK_VIOLATION;
        }
#endif

#ifdef __APPLE__
        // /usr/bin/python3 is a stub on a Mac without the Command Line Tools:
        // running it (even to ask its version) opens their installer. They
        // are installed when `xcode-select -p` succeeds, which is cheap and
        // opens nothing. Asked once per process.
        bool commandLineToolsInstalled() {
            static const bool installed = [] {
                posix_spawn_file_actions_t actions;
                if (::posix_spawn_file_actions_init(&actions) != 0) return false;
                ::posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
                ::posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
                char program[] = "/usr/bin/xcode-select";
                char option[] = "-p";
                char* argv[] = {program, option, nullptr};
                pid_t pid = -1;
                const int rc = ::posix_spawn(&pid, program, &actions, nullptr, argv, environ);
                ::posix_spawn_file_actions_destroy(&actions);
                if (rc != 0) return false;
                int status = 0;
                while (::waitpid(pid, &status, 0) < 0)
                    if (errno != EINTR) return false;
                return WIFEXITED(status) && WEXITSTATUS(status) == 0;
            }();
            return installed;
        }
#endif

        // The directories of $PATH, in order, without empty entries (which
        // would mean the working directory) and without the quotes Windows
        // allows around an entry.
        std::vector<fs::path> pathDirectories() {
#ifdef _WIN32
            const char separator = ';';
#else
            const char separator = ':';
#endif
            std::vector<fs::path> dirs;
            for (std::string d : split(environment("PATH"), separator, true)) {
                if (d.size() >= 2 && d.front() == '"' && d.back() == '"') d = d.substr(1, d.size() - 2);
                if (!d.empty()) dirs.push_back(fsPath(d));
            }
            return dirs;
        }
    } // namespace

    std::string homeDirectory() {
#ifdef _WIN32
        std::string home = wideEnvironment(L"USERPROFILE");
        if (home.empty()) home = wideEnvironment(L"HOMEDRIVE") + wideEnvironment(L"HOMEPATH");
        return forwardSlashes(home);
#else
        const char* home = std::getenv("HOME");
        return home ? std::string(home) : std::string();
#endif
    }

    std::string configDirectory() {
#ifdef _WIN32
        const std::string appdata = wideEnvironment(L"APPDATA");
        return appdata.empty() ? homeDirectory() : forwardSlashes(appdata);
#else
        const char* xdg = std::getenv("XDG_CONFIG_HOME");
        if (xdg && *xdg) return std::string(xdg);
        return homeDirectory() + "/.config";
#endif
    }

    // What lives here holds absolute paths (a venv) and must not roam with
    // the profile, which is why Windows uses the local, not the roaming,
    // application data. A relative $XDG_DATA_HOME is invalid by the XDG
    // specification and is ignored.
    std::string dataDirectory() {
#ifdef _WIN32
        const std::string local = wideEnvironment(L"LOCALAPPDATA");
        if (!local.empty()) return forwardSlashes(local);
        const std::string home = homeDirectory();
        return home.empty() ? std::string() : home + "/AppData/Local";
#else
        const std::string home = homeDirectory();
#ifdef __APPLE__
        return home.empty() ? std::string() : home + "/Library/Application Support";
#else
        const char* xdg = std::getenv("XDG_DATA_HOME");
        if (xdg && xdg[0] == '/') return std::string(xdg);
        return home.empty() ? std::string() : home + "/.local/share";
#endif
#endif
    }

    std::string tempDirectory() {
        std::error_code ec;
        const fs::path p = fs::temp_directory_path(ec);
        return ec ? std::string(".") : forwardSlashes(utf8(p));
    }

    // On POSIX the temporary directory is shared by every user, so a name
    // another user could have created first must not be taken over: mkdtemp
    // picks one that did not exist, creates it with mode 0700 and fails
    // rather than reuse one. %TEMP% on Windows is the user's own; the
    // directory is still a new one, not an existing one taken over.
    std::string makeTempDirectory(const std::string& prefix) {
        std::string parent = tempDirectory();   // Windows' ends in a separator
        if (!parent.empty() && parent.back() != '/') parent += '/';
#ifdef _WIN32
        const std::string base = parent + prefix + std::to_string(processId());
        for (int n = 0; n < 100; ++n) {
            const std::string dir = n == 0 ? base : base + "-" + std::to_string(n);
            const fs::path path = fsPath(dir);
            std::error_code ec;
            if (fs::create_directory(path, ec)) return dir;
            // Left by an earlier process with the same id: try the next name.
            // Anything else (no temporary directory, no permission) will not
            // change with another name.
            if (!fs::exists(path, ec)) return std::string();
        }
        return std::string();
#else
        const std::string pattern = parent + prefix + "XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        return ::mkdtemp(name.data()) ? std::string(name.data()) : std::string();
#endif
    }

    std::string executableDirectory() {
#ifdef _WIN32
        std::wstring buffer(32768, L'\0');
        const DWORD n = ::GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        buffer.resize(n);
        return forwardSlashes(utf8(fs::path(buffer).parent_path()));
#elif defined(__APPLE__)
        std::error_code ec;
        std::uint32_t size = 0;
        ::_NSGetExecutablePath(nullptr, &size);   // fails, and says how much it needs
        std::vector<char> buffer(size + 1, '\0');
        if (::_NSGetExecutablePath(buffer.data(), &size) == 0) {
            // It may name a symbolic link, or hold "." and ".." segments.
            const fs::path p = fs::weakly_canonical(fs::path(buffer.data()), ec);
            if (!ec) return utf8(p.parent_path());
        }
        return utf8(fs::current_path(ec));
#else
        std::error_code ec;
        const fs::path p = fs::read_symlink("/proc/self/exe", ec);
        if (!ec) return utf8(p.parent_path());
        return utf8(fs::current_path(ec));
#endif
    }

    int processId() {
#ifdef _WIN32
        return static_cast<int>(::GetCurrentProcessId());
#else
        return static_cast<int>(::getpid());
#endif
    }

    bool processAlive(int pid) {
        if (pid <= 0) return false;
#ifdef _WIN32
        const HANDLE h = ::OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
        // A process of another user, or a protected one, exists all the same;
        // a pid no process has gives ERROR_INVALID_PARAMETER.
        if (!h) return ::GetLastError() == ERROR_ACCESS_DENIED;
        const bool alive = ::WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
        ::CloseHandle(h);
        return alive;
#else
        if (::kill(static_cast<pid_t>(pid), 0) != 0 && errno != EPERM) return false;
        // A zombie still answers kill(); it has ended, and waits only for its
        // parent (or, in a container, a pid 1 that never reaps) to collect it.
#ifdef __linux__
        std::string stat;
        if (readFile("/proc/" + std::to_string(pid) + "/stat", stat)) {
            const std::size_t close = stat.rfind(')');   // the command name may hold spaces and parentheses
            if (close != std::string::npos && close + 2 < stat.size()) {
                const char state = stat[close + 2];
                if (state == 'Z' || state == 'X') return false;
            }
        }
#elif defined(__APPLE__)
        int query[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
        struct kinfo_proc info{};
        std::size_t size = sizeof info;
        if (::sysctl(query, 4, &info, &size, nullptr, 0) == 0 && size == sizeof info && info.kp_proc.p_stat == SZOMB) return false;
#endif
        return true;
#endif
    }

    std::string environment(const char* name) {
#ifdef _WIN32
        return wideEnvironment(widen(name).c_str());
#else
        const char* v = std::getenv(name);
        return v ? std::string(v) : std::string();
#endif
    }

    bool hasEnvironment(const char* name) {
#ifdef _WIN32
        ::SetLastError(0);
        const DWORD n = ::GetEnvironmentVariableW(widen(name).c_str(), nullptr, 0);
        return n != 0 || ::GetLastError() != ERROR_ENVVAR_NOT_FOUND;
#else
        return std::getenv(name) != nullptr;
#endif
    }

    std::string findPython() {
        const std::vector<fs::path> dirs = pathDirectories();
#ifdef _WIN32
        const std::vector<std::string> names{"python3.exe", "python.exe"};
#else
        const std::vector<std::string> names{"python3", "python"};
#endif
        std::error_code ec;
        const auto usable = [&](const fs::path& p) {
            if (!fs::exists(p, ec)) return false;
#ifdef _WIN32
            if (isStoreAlias(p)) return false;
#endif
#ifdef __APPLE__
            if (p.parent_path() == "/usr/bin" && !commandLineToolsInstalled()) return false;
#endif
            return true;
        };
        for (const std::string& name : names)
            for (const auto& d : dirs)
                if (usable(d / name)) return forwardSlashes(utf8(d / name));
#ifdef _WIN32
        // Directories are walked with increment(error_code): a range-for's
        // operator++ throws when the listing fails half-way.
        //
        // python3.14.exe and the like (uv, the python.org installer without
        // "Add to PATH"): the newest in the first directory that has one, by
        // number (3.13 is newer than 3.9). A free-threaded python3.14t.exe is
        // not taken for the regular build.
        std::string best;
        for (const auto& d : dirs) {
            if (!fs::is_directory(d, ec)) continue;
            int bestMinor = -1;
            std::error_code walk;
            for (fs::directory_iterator it(d, walk), end; !walk && it != end; it.increment(walk)) {
                const fs::path& entry = it->path();
                const std::string file = toLower(utf8(entry.filename()));
                std::string_view rest;
                const int minor = minorVersion(file, "python3.", rest);
                if (minor > bestMinor && rest == ".exe" && usable(entry)) {
                    bestMinor = minor;
                    best = forwardSlashes(utf8(entry));
                }
            }
            if (!best.empty()) return best;
        }
        // The python.org installer's directories (Python313, Python313-32,
        // Python313-arm64): the newest, and of one version the plain one.
        for (const std::string& root : {environment("LOCALAPPDATA") + "/Programs/Python", std::string("C:/Program Files")}) {
            if (!fs::is_directory(fsPath(root), ec)) continue;
            int bestMinor = -1;
            bool bestPlain = false;
            std::error_code walk;
            for (fs::directory_iterator it(fsPath(root), walk), end; !walk && it != end; it.increment(walk)) {
                const fs::path& entry = it->path();
                const std::string file = toLower(utf8(entry.filename()));
                std::string_view rest;
                const int minor = minorVersion(file, "python3", rest);
                const bool plain = rest.empty();
                if (minor < 0 || (!plain && rest.front() != '-')) continue;
                if ((minor > bestMinor || (minor == bestMinor && plain && !bestPlain)) && usable(entry / "python.exe")) {
                    bestMinor = minor;
                    bestPlain = plain;
                    best = forwardSlashes(utf8(entry / "python.exe"));
                }
            }
            if (!best.empty()) return best;
        }
#endif
        return std::string();
    }

    std::string findExecutable(const std::string& name) {
        if (name.empty()) return std::string();
#ifdef _WIN32
        const std::string file = endsWith(toLower(name), ".exe") ? name : name + ".exe";
#else
        const std::string& file = name;
#endif
        const auto usable = [](const fs::path& p) {
            std::error_code ec;
            if (p.empty() || !fs::is_regular_file(p, ec)) return false;
#ifdef _WIN32
            return !isStoreAlias(p);
#else
            return ::access(p.c_str(), X_OK) == 0;
#endif
        };
#ifdef _WIN32
        const bool hasDirectory = file.find_first_of("/\\:") != std::string::npos;
#else
        const bool hasDirectory = file.find('/') != std::string::npos;
#endif
        if (hasDirectory) return usable(fsPath(file)) ? forwardSlashes(file) : std::string();
        for (const fs::path& dir : pathDirectories()) {
            const fs::path candidate = dir / fsPath(file);
            if (usable(candidate)) return forwardSlashes(utf8(candidate));
        }
        return std::string();
    }

    bool isFile(const std::string& path) {
        std::error_code ec;
        return !path.empty() && fs::is_regular_file(fsPath(path), ec);
    }

    bool isDirectory(const std::string& path) {
        std::error_code ec;
        return !path.empty() && fs::is_directory(fsPath(path), ec);
    }

    bool writableByOthers(const std::string& path, std::string* why) {
#ifdef _WIN32
        (void)path;
        (void)why;
        return false;
#else
        struct stat st{};
        if (path.empty() || ::stat(path.c_str(), &st) != 0) return false;
        if (st.st_uid != ::getuid() && st.st_uid != 0) {
            if (why) *why = "it belongs to another user (uid " + std::to_string(st.st_uid) + ")";
            return true;
        }
        if (st.st_mode & S_IWOTH) {
            if (why) *why = "every user may write it";
            return true;
        }
        if (st.st_mode & S_IWGRP) {
            // The user-private-group scheme (umask 002): the user's primary
            // group, named after the user, with no other members, is the
            // user alone.
            const group* g = ::getgrgid(st.st_gid);
            const passwd* pw = ::getpwuid(::getuid());
            const bool privateGroup = g && pw && st.st_gid == ::getgid() && (!g->gr_mem || !g->gr_mem[0]) && g->gr_name && pw->pw_name &&
                                      std::string(g->gr_name) == pw->pw_name;
            if (privateGroup) return false;
            if (why) *why = "its group may write it";
            return true;
        }
        return false;
#endif
    }

    bool makePath(const std::string& dir) {
        const fs::path p = fsPath(dir);
        if (p.empty()) return false;
        std::error_code ec;
        fs::create_directories(p, ec);
        return fs::is_directory(p, ec);
    }

    bool removeTree(const std::string& dir, std::string* error) {
        const fs::path p = fsPath(dir);
        // A slip that makes the path empty or a root ("" + "/", an unset
        // variable) must not become a wiped drive, nor one that makes it the
        // working directory or what holds it ("." from tempDirectory()'s
        // failure). Taken by the letter, after "." and "x/.." are dropped: a
        // path without a name of its own left is refused ("/.", "/tmp/..",
        // "C:.", "..").
        const fs::path normal = p.lexically_normal();
        std::vector<fs::path> names;
        for (const fs::path& part : normal.relative_path()) {
            const bool separator = part.has_root_directory() && !part.has_relative_path() && !part.has_root_name();
            if (!part.empty() && !separator && part != "." && part != "..") names.push_back(part);
        }
        std::size_t needed = 1;
#ifdef _WIN32
        // What is a root on Windows can take more than one name to spell. A
        // share is its server and its name ("//server/share" leaves "share"
        // as a name), and behind a "\\?\", "\\.\" or "\??\" prefix, which is
        // all the root name holds then, come a drive ("\\?\C:\") or "UNC",
        // a server and a share ("\\?\UNC\server\share").
        const std::wstring root = normal.root_name().wstring();
        const auto slash = [](wchar_t c) { return c == L'/' || c == L'\\'; };
        const bool prefix = root.size() == 3 && slash(root[0]) &&
                            ((slash(root[1]) && (root[2] == L'?' || root[2] == L'.')) || (root[1] == L'?' && root[2] == L'?'));
        if (prefix)
            needed = !names.empty() && ::_wcsicmp(names.front().wstring().c_str(), L"UNC") == 0 ? 4 : 2;
        else if (root.size() >= 2 && slash(root[0]) && slash(root[1]))
            needed = 2;
#endif
        if (names.size() < needed) {
            if (error) *error = "refusing to remove '" + dir + "'";
            return false;
        }
        std::error_code ec;
        const auto removed = [&] {
            ec.clear();
            fs::remove_all(p, ec);
            std::error_code ignored;
            return !ec && !fs::exists(fs::symlink_status(p, ignored));
        };
#ifdef _WIN32
        // A file another process has open (a virus scanner reading what pip
        // just wrote) stays behind as "delete pending", and its folder with
        // it, until that process lets go: usually a moment.
        for (int attempt = 1; !removed(); ++attempt) {
            if (attempt >= 40 || (ec && !transient(ec))) {
                if (error) *error = "cannot remove " + forwardSlashes(dir) + ": " + (ec ? ec.message() : std::string("it is still there"));
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        return true;
#else
        if (removed()) return true;
        if (error) *error = "cannot remove " + dir + ": " + (ec ? ec.message() : std::string("it is still there"));
        return false;
#endif
    }

    bool renamePath(const std::string& from, const std::string& to, std::string* error) {
        const fs::path source = fsPath(from), target = fsPath(to);
        if (source.empty() || target.empty()) {
            if (error) *error = "cannot rename '" + from + "' to '" + to + "'";
            return false;
        }
        std::error_code ec;
#ifdef _WIN32
        // Windows refuses to move a folder while any file in it is open, and
        // a virus scanner opens the files that were just written: tried
        // again for about a second, but only for that kind of failure.
        for (int attempt = 1; attempt <= 40; ++attempt) {
            ec.clear();
            fs::rename(source, target, ec);
            if (!ec || !transient(ec)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
#else
        fs::rename(source, target, ec);
#endif
        if (!ec) return true;
        if (error) *error = "cannot rename " + forwardSlashes(from) + " to " + forwardSlashes(to) + ": " + ec.message();
        return false;
    }

    bool readFile(const std::string& path, std::string& out) {
        const fs::path p = fsPath(path);
        if (p.empty()) return false;
        std::ifstream f(p, std::ios::binary);
        if (!f) return false;
        std::ostringstream ss;
        ss << f.rdbuf();
        out = ss.str();
        return true;
    }

    bool writeFileAtomic(const std::string& path, const std::string& content, bool ownerOnly) {
        const fs::path target = fsPath(path);
        if (target.empty()) return false;
        // A name of its own for every call, so two saves at once (from two
        // threads, or two processes) never write into the same file.
        static std::atomic<unsigned> calls{0};
        const auto temporaryName = [&] {
            fs::path name = target;
            name += ".tmp" + std::to_string(processId()) + "-" + std::to_string(calls.fetch_add(1));
            return name;
        };
        fs::path tmp = temporaryName();
#ifdef _WIN32
        (void)ownerOnly;
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) return false;
            f.write(content.data(), static_cast<std::streamsize>(content.size()));
            f.flush();
            if (!f) {
                f.close();
                std::error_code ec;
                fs::remove(tmp, ec);
                return false;
            }
        }
#else
        // Created with its final mode (less the umask), and only under a name
        // nothing has yet. A file made with the default mode and tightened
        // afterwards could be opened by another user in between, who would
        // then read through that descriptor the secret written next; O_EXCL
        // also refuses a link planted under the name. A name that is taken
        // (left behind by a crashed run that had the same pid) is passed
        // over for the next one.
        int fd = -1;
        for (int attempt = 1; (fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, ownerOnly ? 0600 : 0666)) < 0;
             ++attempt) {
            if (errno == EINTR) continue;
            if (errno != EEXIST || attempt >= 100) return false;
            tmp = temporaryName();
        }
        bool written = true;
        std::size_t done = 0;
        while (written && done < content.size()) {
            const ssize_t n = ::write(fd, content.data() + done, content.size() - done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) written = false;
            else done += static_cast<std::size_t>(n);
        }
        // A delayed write error (a full disk, NFS) shows only here.
        if (::close(fd) != 0 && errno != EINTR) written = false;
        if (!written) {
            std::error_code ec;
            fs::remove(tmp, ec);
            return false;
        }
#endif
        // The target is never removed first: a crash, or a rename that keeps
        // failing, must leave the previous content whole. Windows refuses the
        // rename while another handle (a virus scanner, a backup or sync tool)
        // has either file open, usually for moments only, so it is tried again
        // for a while; a rename within a directory elsewhere fails for good.
#ifdef _WIN32
        constexpr int attempts = 10;
#else
        constexpr int attempts = 1;
#endif
        std::error_code ec;
        for (int attempt = 1;; ++attempt) {
            fs::rename(tmp, target, ec);
            if (!ec) return true;
            if (attempt >= attempts) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        std::error_code ignored;
        fs::remove(tmp, ignored);
        return false;
    }

} // namespace sirius::app::host
