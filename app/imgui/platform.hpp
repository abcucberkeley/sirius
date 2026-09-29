#ifndef SIRIUS_IMGUI_PLATFORM_HPP
#define SIRIUS_IMGUI_PLATFORM_HPP

// What the GUI asks of the operating system: where the user's directories
// are, the native file dialogs, opening a URL or a folder, the environment.
// All paths are UTF-8.

#include <string>
#include <vector>

namespace sirius::app::gui::platform {

    std::string homeDirectory();
    std::string configDirectory();      // %APPDATA% / $XDG_CONFIG_HOME / ~/.config
    std::string tempDirectory();
    std::string executableDirectory();  // of the running program
    int processId();
    // "" when the variable is not set.
    std::string environment(const char* name);
    bool hasEnvironment(const char* name);
    // The Python interpreter to start the worker with when nothing names one:
    // the first of python3 / python on PATH (on Windows also python3.x.exe and
    // the py launcher's installs), skipping the Microsoft Store alias that
    // only prints how to install Python. "" when there is none.
    std::string findPython();
    // Creates the directory and its parents; true when it exists afterwards.
    bool makePath(const std::string& dir);
    // Reads / writes a whole file; read returns false when it cannot be opened.
    bool readFile(const std::string& path, std::string& out);
    // Written beside the file and renamed over it, so a crash leaves the
    // previous content whole. False when it could not be written; the
    // previous content is then left as it was.
    bool writeFileAtomic(const std::string& path, const std::string& content, bool ownerOnly = false);

    // --- native dialogs (blocking; the window waits as under a modal dialog) ---
    struct FileFilter {
        std::string name;         // "TIFF"
        std::string extensions;   // "tif,tiff" (no dots, comma separated); "*" = all files
    };
    // A parameter's filter string as filters: entries separated by ";;", each a
    // name and its patterns in parentheses, "SIRIUS pipeline (*.sirius.toml *.toml);;All files (*)".
    // Compound extensions keep their last part ("sirius.toml" -> "toml").
    std::vector<FileFilter> parseFileFilters(const std::string& filter);

    // Empty when cancelled, and also when the dialog could not be opened:
    // takeDialogError() tells the two apart. `start` may be a directory or a
    // file in it. The dialog belongs to the application window it is opened
    // from (the main window outside a frame), unless that one is minimised;
    // the clicks and keys the windows received meanwhile are dropped when it
    // returns, rather than replayed into the next frame.
    std::string openFileDialog(const std::string& title, const std::string& start,
                               const std::vector<FileFilter>& filters = {});
    std::vector<std::string> openFilesDialog(const std::string& title, const std::string& start,
                                             const std::vector<FileFilter>& filters = {});
    std::string saveFileDialog(const std::string& title, const std::string& directory, const std::string& defaultName,
                               const std::vector<FileFilter>& filters = {});
    std::string pickFolderDialog(const std::string& title, const std::string& start);
    // Why the last dialog could not be opened, as a sentence to show the
    // user; taking it clears it. Empty after a choice or a cancel.
    std::string takeDialogError();
    // The frame loop says when it is inside GLFW's event processing
    // (glfwPollEvents, glfwWaitEvents*), where a window refresh callback may
    // draw a frame that opens a dialog. GLFW must not be polled from its
    // callbacks, so such a dialog drops only the input already delivered.
    void setWithinEventPoll(bool within);

    // The default browser / file manager.
    void openUrl(const std::string& url);
    void openInFileManager(const std::string& path);

    // A console the program was started from (Windows GUI subsystem): stdout
    // and stderr are attached to it so --help and the scripting output print.
    void attachParentConsole();

} // namespace sirius::app::gui::platform

#endif // SIRIUS_IMGUI_PLATFORM_HPP
