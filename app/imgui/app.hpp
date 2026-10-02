#ifndef SIRIUS_IMGUI_APP_HPP
#define SIRIUS_IMGUI_APP_HPP

// The window shell of docs/design: title/menu bar (brand, seven menus,
// dataset · GPU readout, ✦ Assistant), the Operations / Parameters / viewer /
// Diagnostics / Log / Assistant docks (each moved by its tab, and docked,
// floated or maximised by the controls on its tab bar), the status bar,
// every menu action and keyboard shortcut, layout persistence, the file
// dialogs and the message boxes. All state lives in the Workbench; this
// class only routes.
//
// One frame:
//     bridge.update()      posted functions, run progress, finished jobs
//     deferred actions     what panels asked to happen outside a window
//     ImGui frame          title bar, dock windows -> panel.draw(), dialogs
//     render + swap
//
// Panels never block. A modal dialog is a Dialog: showDialog() puts it on
// screen, the frame loop keeps running, and the dialog calls back when it
// is accepted. ask() and message() are the
// message boxes, answered through a callback the same way.
//
// Anything that changes what is on screen in a way Dear ImGui must not see
// mid-frame (opening a native file dialog, waiting for a run) goes through
// defer(): it runs between two frames. Only there may waitUntil() be used,
// which keeps drawing frames until its condition holds -- how a scripted
// "run" waits for the worker thread without freezing the window.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>

#include "core/tool_api.hpp"
#include "core/workbench.hpp"
#include "imgui/bridge.hpp"
#include "imgui/worker_launcher.hpp"

struct GLFWwindow;

namespace sirius::app::gui {

    class App;
    class Viewer;
    class OpsPanel;
    class ParamsPanel;
    class DiagnosticsPanel;
    class AssistantPanel;
    class LogPanel;
    class HelpWindow;
    class ClusterLink;

    // A dialog: a titled window over the application, modal unless it says
    // otherwise. The application draws the frame, the title row and the
    // close box; draw() fills the body and calls close() when it is done
    // (after calling whatever callback the dialog was given).
    class Dialog {
    public:
        virtual ~Dialog() = default;
        virtual std::string title() const = 0;
        // Design pixels; a height of 0 fits the content.
        virtual ImVec2 size() const { return ImVec2(640, 0); }
        virtual bool modal() const { return true; }
        virtual bool resizable() const { return false; }
        virtual void draw(App& app) = 0;
        // The close box or Escape: false keeps the dialog open (unsaved text).
        virtual bool canClose(App&) { return true; }
        // Called once when the dialog leaves the screen, however it was closed.
        virtual void closed(App&) {}

        void close() { open_ = false; }
        bool isOpen() const noexcept { return open_; }
        // Bring an already open (non-modal) dialog to the front next frame.
        void raise() { raise_ = true; }
        // No modal dialog is open over this one: the keys (Enter) are its own.
        bool onTop() const noexcept { return onTop_; }

    private:
        friend class App;
        bool open_ = true;
        bool raise_ = false;
        bool appeared_ = false;
        bool onTop_ = true;
    };

    enum class MessageIcon { None,
                             Info,
                             Warning,
                             Question };

    struct InitOptions {
        bool unattended = false;      // scripting / screenshots: nobody answers a question
        bool visible = true;          // false: the window is never shown (headless grabs)
        int width = 1600, height = 960;
        bool sizeGiven = false;       // --size: overrides the saved window size and maximized state
    };

    class App {
    public:
        App(Bridge& bridge, ToolApi& tools, WorkerLauncher& launcher);
        ~App();
        App(const App&) = delete;
        App& operator=(const App&) = delete;

        // Creates the window, the OpenGL context and the Dear ImGui context,
        // loads the fonts and the layout. False (message on stderr) when the
        // platform has no display or no OpenGL 3.3.
        bool init(const InitOptions& options);
        // Frames until the window closes; returns the process's exit code.
        int run();
        // One frame. False once the window is closing.
        bool frame();
        void requestClose();          // asks about unsaved work first, unless unattended
        void quitNow();               // no questions
        // The window is closing: the frame loop ends after this frame.
        bool closing() const;
        // Draw this many frames without waiting for input (animation, results arriving).
        void requestRedraw(int frames = 3);

        Bridge& bridge() noexcept { return bridge_; }
        Workbench& wb() noexcept { return bridge_.wb(); }
        ToolApi& tools() noexcept { return tools_; }
        WorkerLauncher& launcher() noexcept { return launcher_; }
        GLFWwindow* window() const noexcept;
        bool unattended() const noexcept;
        void setUnattended(bool on);
        void setExitCode(int code);
        int exitCode() const noexcept;

        // --- between frames --------------------------------------------------
        void defer(std::function<void()> action);
        // Draws frames until `done()` holds or the window closes. Only inside
        // a deferred action (or before run()).
        void waitUntil(const std::function<bool()>& done);
        // Lets the window react, and a dataset still loading arrive, between
        // two scripted steps.
        void settle();

        // --- dialogs and boxes --------------------------------------------------
        void showDialog(std::shared_ptr<Dialog> dialog);
        bool dialogOpen() const;      // a modal one
        // An OK box that does not hold up whoever raised it.
        void message(const std::string& title, const std::string& text, MessageIcon icon = MessageIcon::Warning);
        // A question; `answer` receives the index of the button pressed, or -1
        // when the box was dismissed (Escape, the close box). The last button
        // is the default. Unattended: answered at once with `unattendedAnswer`.
        void ask(const std::string& title, const std::string& text, const std::vector<std::string>& buttons,
                 std::function<void(int)> answer, int unattendedAnswer = -1);
        // One line of text; `accepted` is not called when cancelled.
        // `password` hides what is typed (tokens, keys).
        void promptText(const std::string& title, const std::string& label, const std::string& initial,
                        std::function<void(const std::string&)> accepted, bool password = false);

        // --- commands: datasets and pipelines ---------------------------------
        void openDatasetDialog();
        void openFolderDataset();
        // A plain folder of files goes through the pattern dialog that builds
        // the manifest; everything else opens directly.
        void openDatasetPath(const std::string& path);
        void openWith(const std::string& path, OpenOptions options);
        void openPipelinePath(const std::string& path);
        // As though these paths were dropped on the window.
        void dropPaths(const std::vector<std::string>& paths);
        void savePipelineTo(const std::string& path);   // "" asks
        void loadPipeline();
        void toggleRecording();
        // Recent datasets, newest first (the setting "recent/datasets").
        static std::vector<std::string> recentFiles();
        static void addRecentFile(const std::string& path);

        // --- commands: export ------------------------------------------------------
        void exportResultDialog();
        void exportTrainingDialog();
        void exportPythonScript();
        void exportFigureImage();
        void exportLabels();

        // --- commands: steps ---------------------------------------------------------
        // Removes with the evidence in the log; asks first when the step
        // holds a large computed output.
        void removeStepAt(int index);
        // The step on its own: refuses, out loud, when its input is missing.
        void runSelectedStep();
        void runAll();
        void runTo(int index);
        void cancel();                // the run and the task
        int segmentationStep() const;
        int segmentationStepOrNew();
        int stepOrNew(const std::string& kind);
        void loadTorchModel();
        void modelHub();
        void mergeLabelsDialog();
        void selectFlagged(bool forward);

        // --- commands: windows ---------------------------------------------------------
        void preferences();
        // Process ▸ Connect to cluster…, and the session behind it.
        void clusterDialog();
        ClusterLink& cluster();
        void pluginManager(const std::string& file = {});
        void showHelpForSelected();
        void showHelp(const std::string& kind);
        void toggleHelp();
        bool helpOpen() const;
        void showLog();
        void showOperations();
        void setAssistantVisible(bool on);
        bool assistantVisible() const;
        void askAssistant(const std::string& text);
        void setDiagnosticsMaximized(bool on);
        bool diagnosticsMaximized() const;
        void floatDiagnostics();
        bool diagnosticsFloating() const;
        void dockDiagnostics();
        void resetLayout();
        void about();

        // The directory the file dialogs start in.
        std::string lastDir() const;
        void setLastDir(const std::string& dir);

        // --- panels --------------------------------------------------------------------
        Viewer& viewer();
        OpsPanel& ops();
        ParamsPanel& params();
        DiagnosticsPanel& diagnostics();
        AssistantPanel& assistant();
        LogPanel& log();
        HelpWindow& help();

        // --- scripting ---------------------------------------------------------------------
        // Triggers a menu action by its text ("Export result", with or without
        // the ellipsis); false when there is none.
        bool triggerAction(const std::string& text);
        // The window as it was last drawn, to a PNG.
        bool screenshot(const std::string& path);
        // A panel that uses a key chord itself this frame (Ctrl+C in the log)
        // claims it, and the menu action bound to the same chord stands back.
        // Claimed while drawing; honoured from the next frame on, so a panel
        // claims the chord on every frame it wants it (while it has focus).
        void claimKey(ImGuiKeyChord chord);
        // Frames drawn since the start.
        std::uint64_t frameCount() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        Bridge& bridge_;
        ToolApi& tools_;
        WorkerLauncher& launcher_;
    };

    // The shortcut as this platform writes it: "Ctrl+Shift+R", "Alt+Up", "F1".
    std::string shortcutText(ImGuiKeyChord chord);

    // The shortcuts a panel has to name in a tool tip, in one place.
    namespace keys {
        inline constexpr ImGuiKeyChord runAll = ImGuiMod_Ctrl | ImGuiKey_R;
        inline constexpr ImGuiKeyChord runSelected = ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_R;
        inline constexpr ImGuiKeyChord removeStep = ImGuiKey_Backspace;
        inline constexpr ImGuiKeyChord duplicateStep = ImGuiMod_Ctrl | ImGuiKey_D;
        inline constexpr ImGuiKeyChord enableStep = ImGuiKey_Space;
        inline constexpr ImGuiKeyChord moveUp = ImGuiMod_Alt | ImGuiKey_UpArrow;
        inline constexpr ImGuiKeyChord moveDown = ImGuiMod_Alt | ImGuiKey_DownArrow;
        inline constexpr ImGuiKeyChord undo = ImGuiMod_Ctrl | ImGuiKey_Z;
        inline constexpr ImGuiKeyChord helpForStep = ImGuiKey_F1;
        inline constexpr ImGuiKeyChord assistant = ImGuiMod_Alt | ImGuiKey_5;
        inline constexpr ImGuiKeyChord logDock = ImGuiMod_Alt | ImGuiKey_7;
        inline constexpr ImGuiKeyChord send = ImGuiKey_Enter;
    } // namespace keys

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_APP_HPP
