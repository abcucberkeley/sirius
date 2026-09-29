#ifndef SIRIUS_APP_HOST_HPP
#define SIRIUS_APP_HOST_HPP

// What the application asks of the operating system, without a window: the user's
// directories, the environment, executables on PATH, whole files. UTF-8 paths with
// forward slashes. Moved out of imgui/platform (which re-exports these).
//
// GUI-free and thread-safe: the GUI, sirius-cli and the Python environment's
// setup share it. Nothing here throws; a failure is an empty string or false.

#include <string>

namespace sirius::app::host {
    std::string homeDirectory();
    std::string configDirectory();                  // %APPDATA% | $XDG_CONFIG_HOME | ~/.config
    std::string dataDirectory();                    // %LOCALAPPDATA% | $XDG_DATA_HOME (absolute) or ~/.local/share | ~/Library/Application Support
    std::string tempDirectory();
    std::string makeTempDirectory(const std::string& prefix);   // a new directory of this process's own; "" when none
    std::string executableDirectory();              // GetModuleFileNameW | /proc/self/exe | _NSGetExecutablePath
    int processId();
    // False for a pid no process has (or that only a zombie holds); true for
    // one that runs, also when it belongs to a user this process may not
    // signal or open.
    bool processAlive(int pid);
    std::string environment(const char* name);      // "" when unset
    bool hasEnvironment(const char* name);
    // The Python interpreter to start the worker with when nothing names one:
    // the first of python3 / python on PATH (on Windows also python3.x.exe and
    // the python.org installer's directories), skipping the Microsoft Store
    // alias that only prints how to install Python, and on macOS
    // /usr/bin/python3 while the Command Line Tools it stands in for are not
    // installed. "" when there is none.
    std::string findPython();                       // today's platform::findPython rule; macOS skips the Xcode stub
    // A name with a directory is only checked. On POSIX the file must be
    // executable; on Windows ".exe" is added unless the name ends in it.
    std::string findExecutable(const std::string& name);   // PATH (+".exe" on Windows), no WindowsApps aliases; "" when none
    bool isFile(const std::string& path);
    bool isDirectory(const std::string& path);
    // Creates the directory and its parents; true when it exists afterwards.
    bool makePath(const std::string& dir);
    // True when `dir` is gone afterwards, also when it never existed. An empty
    // path is refused, and so is one that is left without a name of its own
    // once "." and "x/.." are dropped: a root ("/.", "C:."), or the working
    // directory or one above it (".", ".."). On Windows a file another
    // process holds open for a moment (a virus scanner) is waited for, about
    // a second.
    bool removeTree(const std::string& dir, std::string* error = nullptr);
    bool renamePath(const std::string& from, const std::string& to, std::string* error = nullptr);   // retried ~1 s on Windows
    // Reads a whole file; false when it cannot be opened.
    bool readFile(const std::string& path, std::string& out);
    // Written beside the file and renamed over it, so a crash leaves the
    // previous content whole. False when it could not be written; the
    // previous content is then left as it was. `ownerOnly` makes the file
    // readable by its owner only from the moment it is created (POSIX; on
    // Windows the user's profile directories already are).
    bool writeFileAtomic(const std::string& path, const std::string& content, bool ownerOnly = false);
} // namespace sirius::app::host

#endif // SIRIUS_APP_HOST_HPP
