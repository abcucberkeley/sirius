#include "imgui/platform.hpp"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <system_error>

#include <nfd.h>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#else
#include <cerrno>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "imgui/native_dialog.hpp"
#include "imgui/strings.hpp"

namespace sirius::app::gui::platform {

    namespace {
        std::filesystem::path fsPath(const std::string& p) { return std::filesystem::u8path(p); }

#ifdef _WIN32
        std::wstring widen(const std::string& s) {
            if (s.empty()) return std::wstring();
            const int n = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
            std::wstring w(static_cast<std::size_t>(n), L'\0');
            ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
            return w;
        }
#endif

        // Windows paths as the rest of the application writes them. Elsewhere
        // a backslash is an ordinary character of a file name, left alone.
        std::string forwardSlashes(std::string s) {
#ifdef _WIN32
            for (char& c : s)
                if (c == '\\') c = '/';
#endif
            return s;
        }

        // Why the last dialog could not be opened (takeDialogError). Dialogs
        // run on the main thread only, like everything that touches the window.
        std::string& dialogError() {
            static std::string error;
            return error;
        }

        // Inside the frame loop's GLFW event processing (setWithinEventPoll).
        bool withinEventPoll = false;

        // Keeps NFD's reason for an NFD_ERROR, which callers would otherwise
        // take for a cancel, and prints it. Clearing it also frees what the
        // portal backend holds for it.
        void noteNfdError() {
            const char* error = NFD_GetError();
            std::string reason = error ? error : "";
            while (!reason.empty() && (reason.back() == '.' || reason.back() == ' ')) reason.pop_back();
            std::string text = "The file dialog could not be opened: " + (reason.empty() ? std::string("unknown error") : reason) + ".";
#ifdef NFD_PORTAL
            text += " It goes through xdg-desktop-portal, which needs a FileChooser backend such as "
                    "xdg-desktop-portal-gtk (a build with libgtk-3-dev installed uses GTK's dialog instead).";
#endif
            std::fprintf(stderr, "sirius-app: %s\n", text.c_str());
            dialogError() = std::move(text);
            NFD_ClearError();
        }

        // NFD is initialised on first use and torn down at exit. A failed
        // initialisation (no session bus yet, say) is tried again by the next
        // dialog rather than remembered.
        bool ensureNfd() {
            static bool initialised = false;
            if (initialised) return true;
            // GTK's initialisation sets the whole C locale from the environment;
            // numbers must go on being written and read with a '.'.
            const char* before = std::setlocale(LC_NUMERIC, nullptr);
            const std::string numeric = before ? before : "C";
            const nfdresult_t result = NFD_Init();
            const char* after = std::setlocale(LC_NUMERIC, nullptr);
            if (after == nullptr || numeric != after) std::setlocale(LC_NUMERIC, numeric.c_str());
            if (result != NFD_OKAY) {
                noteNfdError();
                return false;
            }
            initialised = true;
            std::atexit([] { NFD_Quit(); });
            return true;
        }

        // Maps what a dialog returned: true when something was chosen. A
        // cancel is no error; NFD_ERROR is noted for takeDialogError.
        bool chosen(nfdresult_t result) {
            if (result == NFD_ERROR) noteNfdError();
            return result == NFD_OKAY;
        }

        // Around each dialog: a fresh error, the parent window, and afterwards
        // the input the application received while it was open dropped.
        struct DialogScope {
            nfdwindowhandle_t parent{};
            DialogScope() {
                dialogError().clear();
                parent = native_dialog::parentWindow();
            }
            ~DialogScope() { native_dialog::dropQueuedInput(!withinEventPoll); }
            DialogScope(const DialogScope&) = delete;
            DialogScope& operator=(const DialogScope&) = delete;
        };

        // Absolute and without "." or ".." segments: the Windows shell rejects
        // a start folder such as "..\out" (and NFD then opens no dialog), and
        // the portal would resolve a relative one against its own directory.
        std::string startDirectory(const std::string& start) {
            if (start.empty()) return std::string();
            std::error_code ec;
            std::filesystem::path p = fsPath(absolutePath(start));
            if (std::filesystem::is_regular_file(p, ec)) p = p.parent_path();
            while (!p.empty() && !std::filesystem::is_directory(p, ec)) {
                const std::filesystem::path parent = p.parent_path();
                if (parent == p) return std::string();
                p = parent;
            }
            return p.u8string();
        }

        struct NfdFilters {
            std::vector<nfdu8filteritem_t> items;
            explicit NfdFilters(const std::vector<FileFilter>& filters) {
                for (const FileFilter& f : filters) {
                    if (f.extensions.empty() || f.extensions == "*") continue;   // NFD always offers "all files"
                    items.push_back({f.name.c_str(), f.extensions.c_str()});
                }
            }
            const nfdu8filteritem_t* data() const { return items.empty() ? nullptr : items.data(); }
            nfdfiltersize_t size() const { return static_cast<nfdfiltersize_t>(items.size()); }
        };
    } // namespace

    std::vector<FileFilter> parseFileFilters(const std::string& filter) {
        std::vector<FileFilter> out;
        for (const std::string& entry : split(replaceAll(filter, ";;", "\n"), '\n', true)) {
            const std::size_t open = entry.find('('), close = entry.rfind(')');
            if (open == std::string::npos || close == std::string::npos || close < open) continue;
            FileFilter f;
            f.name = trimmed(entry.substr(0, open));
            std::vector<std::string> extensions;
            for (const std::string& pattern : split(entry.substr(open + 1, close - open - 1), ' ', true)) {
                if (pattern == "*" || pattern == "*.*") {
                    extensions.clear();
                    extensions.push_back("*");
                    break;
                }
                const std::size_t dot = pattern.rfind('.');
                const std::string ext = dot == std::string::npos ? pattern : pattern.substr(dot + 1);
                if (!ext.empty() && std::find(extensions.begin(), extensions.end(), ext) == extensions.end())
                    extensions.push_back(ext);
            }
            f.extensions = join(extensions, ",");
            out.push_back(std::move(f));
        }
        return out;
    }

    std::string openFileDialog(const std::string& /*title*/, const std::string& start,
                               const std::vector<FileFilter>& filters) {
        const DialogScope scope;
        if (!ensureNfd()) return std::string();
        const NfdFilters f(filters);
        const std::string dir = startDirectory(start);
        nfdopendialogu8args_t args{};
        args.filterList = f.data();
        args.filterCount = f.size();
        args.defaultPath = dir.empty() ? nullptr : dir.c_str();
        args.parentWindow = scope.parent;
        nfdu8char_t* out = nullptr;
        if (!chosen(NFD_OpenDialogU8_With(&out, &args))) return std::string();
        std::string path = forwardSlashes(out);
        NFD_FreePathU8(out);
        return path;
    }

    std::vector<std::string> openFilesDialog(const std::string& /*title*/, const std::string& start,
                                             const std::vector<FileFilter>& filters) {
        const DialogScope scope;
        std::vector<std::string> paths;
        if (!ensureNfd()) return paths;
        const NfdFilters f(filters);
        const std::string dir = startDirectory(start);
        nfdopendialogu8args_t args{};
        args.filterList = f.data();
        args.filterCount = f.size();
        args.defaultPath = dir.empty() ? nullptr : dir.c_str();
        args.parentWindow = scope.parent;
        const nfdpathset_t* set = nullptr;
        if (!chosen(NFD_OpenDialogMultipleU8_With(&set, &args))) return paths;
        nfdpathsetsize_t n = 0;
        NFD_PathSet_GetCount(set, &n);
        for (nfdpathsetsize_t i = 0; i < n; ++i) {
            nfdu8char_t* p = nullptr;
            if (NFD_PathSet_GetPathU8(set, i, &p) == NFD_OKAY) {
                paths.push_back(forwardSlashes(p));
                NFD_PathSet_FreePathU8(p);
            }
        }
        NFD_PathSet_Free(set);
        return paths;
    }

    std::string saveFileDialog(const std::string& /*title*/, const std::string& directory, const std::string& defaultName,
                               const std::vector<FileFilter>& filters) {
        const DialogScope scope;
        if (!ensureNfd()) return std::string();
        const NfdFilters f(filters);
        const std::string dir = startDirectory(directory);
        nfdsavedialogu8args_t args{};
        args.filterList = f.data();
        args.filterCount = f.size();
        args.defaultPath = dir.empty() ? nullptr : dir.c_str();
        args.defaultName = defaultName.empty() ? nullptr : defaultName.c_str();
        args.parentWindow = scope.parent;
        nfdu8char_t* out = nullptr;
        if (!chosen(NFD_SaveDialogU8_With(&out, &args))) return std::string();
        std::string path = forwardSlashes(out);
        NFD_FreePathU8(out);
        return path;
    }

    std::string pickFolderDialog(const std::string& /*title*/, const std::string& start) {
        const DialogScope scope;
        if (!ensureNfd()) return std::string();
        const std::string dir = startDirectory(start);
        nfdpickfolderu8args_t args{};
        args.defaultPath = dir.empty() ? nullptr : dir.c_str();
        args.parentWindow = scope.parent;
        nfdu8char_t* out = nullptr;
        if (!chosen(NFD_PickFolderU8_With(&out, &args))) return std::string();
        std::string path = forwardSlashes(out);
        NFD_FreePathU8(out);
        return path;
    }

    std::string takeDialogError() {
        std::string error;
        error.swap(dialogError());
        return error;
    }

    void setWithinEventPoll(bool within) { withinEventPoll = within; }

    void openUrl(const std::string& url) {
#ifdef _WIN32
        ::ShellExecuteW(nullptr, L"open", widen(url).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
        // fork + exec rather than system(): the URL never meets a shell. Twice:
        // the child exits at once and is reaped here, and the grandchild that
        // runs the opener is left to init, so neither lingers as a zombie.
        const pid_t child = ::fork();
        if (child == 0) {
            if (::fork() == 0) {
                ::setsid();   // not stopped with the terminal sirius-app was started from
                // The application ignores SIGPIPE (main.cpp), and an ignored
                // signal stays ignored across exec: the opener and the browser
                // it starts get the default back.
                ::signal(SIGPIPE, SIG_DFL);
#ifdef __APPLE__
                ::execlp("open", "open", url.c_str(), static_cast<char*>(nullptr));
#else
                ::execlp("xdg-open", "xdg-open", url.c_str(), static_cast<char*>(nullptr));
#endif
                ::_exit(127);
            }
            ::_exit(0);
        }
        if (child > 0) {
            int status = 0;
            while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
            }
        }
#endif
    }

    void openInFileManager(const std::string& path) {
#ifdef _WIN32
        std::string native = path;
        for (char& c : native)
            if (c == '/') c = '\\';
        ::ShellExecuteW(nullptr, L"open", widen(native).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
        openUrl(path);
#endif
    }

    void attachParentConsole() {
#ifdef _WIN32
        if (!::AttachConsole(ATTACH_PARENT_PROCESS)) return;
        // only the streams that are not already redirected to a file or a pipe
        const auto detached = [](DWORD which) {
            const HANDLE h = ::GetStdHandle(which);
            return h == nullptr || h == INVALID_HANDLE_VALUE || ::GetFileType(h) == FILE_TYPE_UNKNOWN;
        };
        FILE* f = nullptr;
        if (detached(STD_OUTPUT_HANDLE)) ::freopen_s(&f, "CONOUT$", "w", stdout);
        if (detached(STD_ERROR_HANDLE)) ::freopen_s(&f, "CONOUT$", "w", stderr);
#endif
    }

} // namespace sirius::app::gui::platform
