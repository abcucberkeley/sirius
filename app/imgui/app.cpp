#include "imgui/app.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <utility>

#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include "imgui/gl.hpp"

#include <GLFW/glfw3.h>

#include <sirius/device.hpp>
#include <sirius/tiff_io.hpp>

#include "core/app_paths.hpp"
#include "core/array_source.hpp"
#include "core/export.hpp"
#include "core/training_export.hpp"
#include "imgui/cluster_link.hpp"
#include "imgui/dialogs/dialogs.hpp"
#include "imgui/http.hpp"
#include "imgui/panels/assistant_panel.hpp"
#include "imgui/panels/diagnostics_panel.hpp"
#include "imgui/panels/help_window.hpp"
#include "imgui/panels/log_panel.hpp"
#include "imgui/panels/ops_panel.hpp"
#include "imgui/panels/params_panel.hpp"
#include "imgui/platform.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/viewer/viewer.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    using theme::px;
    using theme::Weight;

    namespace {

        constexpr const char* kIssuesUrl = "https://github.com/abcucberkeley/sirius/issues";
        constexpr const char* kRepositoryUrl = "https://github.com/abcucberkeley/sirius";
        constexpr int kMaxRecent = 12;

        // window names: what Dear ImGui keys the layout by
        constexpr const char* kOpsWindow = "Operations";
        constexpr const char* kParamsWindow = "Parameters";
        constexpr const char* kDiagWindow = "Diagnostics";
        constexpr const char* kLogWindow = "Log";
        constexpr const char* kAssistantWindow = "Assistant";
        constexpr const char* kViewerWindow = "Viewer";
        // A maximised panel's stand-in window: its name plus this, so its tab
        // reads the same.
        constexpr const char* kStandInSuffix = "##maximized";

        // The dock windows, in the order they are drawn.
        enum class Panel { Viewer,
                           Ops,
                           Params,
                           Assistant,
                           Log,
                           Diag };
        constexpr std::size_t kPanelCount = 6;

        // What the Float and Dock controls ask of a panel's window at its next Begin.
        enum class Move { None,
                          Float,
                          Dock };

        using Clock = std::chrono::steady_clock;

        double secondsSince(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

        std::string deviceLabel(int i) {
            try {
                const DeviceProperties p = deviceProperties(Device::cuda(i));
                return format("cuda:%d \xC2\xB7 %s \xC2\xB7 %.1f GB", i, p.name.c_str(),
                              static_cast<double>(p.totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0));
            } catch (const std::exception&) {
                return format("cuda:%d", i);
            }
        }

        const char* keyName(ImGuiKey key) {
            switch (key) {
                case ImGuiKey_UpArrow: return "Up";
                case ImGuiKey_DownArrow: return "Down";
                case ImGuiKey_LeftArrow: return "Left";
                case ImGuiKey_RightArrow: return "Right";
                case ImGuiKey_Equal: return "+";
                case ImGuiKey_KeypadAdd: return "+";
                case ImGuiKey_Minus: return "-";
                case ImGuiKey_KeypadSubtract: return "-";
                case ImGuiKey_Slash: return "/";
                case ImGuiKey_Comma: return ",";
                case ImGuiKey_Escape: return "Esc";
                case ImGuiKey_Backspace: return "Backspace";
                case ImGuiKey_Delete: return "Del";
                case ImGuiKey_Space: return "Space";
                case ImGuiKey_Enter: return "Enter";
                default: return ImGui::GetKeyName(key);
            }
        }

        // What a dropped path is, by extension and by what is inside a folder.
        enum class DropKind { None,
                              Dataset,
                              Folder,
                              Pipeline,
                              Plugin };

        DropKind kindOfDrop(const std::string& path) {
            if (isDirectory(path)) return DropKind::Folder;
            if (!isFile(path)) return DropKind::None;
            const std::string name = toLower(fileName(path));
            if (endsWith(name, ".sirius.toml")) return DropKind::Pipeline;
            if (isDatasetManifestFile(path)) return DropKind::Dataset;
            // Any other TOML is a pipeline, as Load pipeline and the command
            // line take it: a save dialog that completed a bare name gives "x.toml".
            if (endsWith(name, ".toml")) return DropKind::Pipeline;
            if (endsWith(name, ".py")) return DropKind::Plugin;
            for (const char* ext : {".tif", ".tiff", ".ome.tif", ".ome.tiff", ".zarr", ".n5", ".sir5"})
                if (endsWith(name, ext)) return DropKind::Dataset;
            return DropKind::None;
        }

        // --- scale and monitors ------------------------------------------------

        // Where the framebuffer carries the pixel density (Wayland, macOS),
        // window coordinates are already logical: the content scale must not be
        // applied to them again. Asked of GLFW itself, so it holds before the
        // Dear ImGui backend is initialised.
        bool framebufferCarriesScale() {
#ifdef __APPLE__
            return true;
#else
            return glfwGetPlatform() == GLFW_PLATFORM_WAYLAND;
#endif
        }

        // Window coordinates per design pixel on this monitor, or in this
        // window: the content scale, or 1 where the framebuffer carries it.
        float monitorScale(GLFWmonitor* monitor) {
            float xs = 1.0f, ys = 1.0f;
            if (monitor && !framebufferCarriesScale()) glfwGetMonitorContentScale(monitor, &xs, &ys);
            return std::max(xs, 0.5f);
        }

        float windowScale(GLFWwindow* window) {
            float xs = 1.0f, ys = 1.0f;
            if (window && !framebufferCarriesScale()) glfwGetWindowContentScale(window, &xs, &ys);
            return std::max(xs, 0.5f);
        }

        // Floating windows are drawn at the main window's scale, so they only
        // leave it when every monitor has that scale.
        bool monitorsShareScale() {
            int count = 0;
            GLFWmonitor** monitors = glfwGetMonitors(&count);
            for (int i = 1; i < count; ++i)
                if (std::abs(monitorScale(monitors[i]) - monitorScale(monitors[0])) > 0.01f) return false;
            return true;
        }

        // --- docks -------------------------------------------------------------

        constexpr ImGuiWindowFlags kPanelFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

        // Every panel keeps a tab bar, floating as well as docked: its tab is
        // how it is moved, and its tab bar holds its window controls. So a
        // floating panel is a dock of its own, and Dear ImGui's dock window
        // (with the tab bar, no title bar) is the floating window.
        const ImGuiWindowClass& panelClass() {
            static const ImGuiWindowClass c = [] {
                ImGuiWindowClass k;
                k.DockingAlwaysTabBar = true;
                return k;
            }();
            return c;
        }

        // A maximised panel's stand-in: a tab bar too, and a class of its own,
        // so it docks into nothing and nothing docks into it (not even another
        // stand-in), and its tab cannot be dragged out of its dock.
        const ImGuiWindowClass& standInClass() {
            static const ImGuiWindowClass c = [] {
                ImGuiWindowClass k;
                k.ClassId = ImHashStr("SiriusMaximizedPanel");
                k.DockingAllowUnclassed = false;
                k.DockingAlwaysTabBar = true;
                k.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoDockingOverMe | ImGuiDockNodeFlags_NoUndocking;
                return k;
            }();
            return c;
        }

        // Whether a dock node is part of the main dockspace, rather than of a
        // group floating on its own (possibly on another monitor).
        bool inDockspace(const ImGuiDockNode* node, ImGuiID dockspace) {
            while (node && node->ParentNode) node = node->ParentNode;
            return node && node->ID == dockspace;
        }

        // The central node of the dock tree under `node`, found by walking the
        // tree. Not DockBuilderGetCentralNode: that reads a pointer the root
        // refreshes only when the dockspace is laid out, and a merge since then
        // (the last panel floated out of a dock beside the central node) may
        // have freed the node it points to.
        ImGuiDockNode* centralNode(ImGuiDockNode* node) {
            if (!node) return nullptr;
            if (node->IsLeafNode()) return node->IsCentralNode() ? node : nullptr;
            if (ImGuiDockNode* found = centralNode(node->ChildNodes[0])) return found;
            return centralNode(node->ChildNodes[1]);
        }

        // The docks' sizes are display pixels: scaled with the design pixels,
        // the side docks keep their design widths on a monitor of another scale.
        void scaleDockTree(ImGuiDockNode* node, float ratio) {
            if (!node) return;
            node->Size = ImVec2(std::round(node->Size.x * ratio), std::round(node->Size.y * ratio));
            node->SizeRef = ImVec2(std::round(node->SizeRef.x * ratio), std::round(node->SizeRef.y * ratio));
            scaleDockTree(node->ChildNodes[0], ratio);
            scaleDockTree(node->ChildNodes[1], ratio);
        }

        // A layout saved when the docks hid their tabs keeps them hidden: the
        // dockspace's AutoHideTabBar left HiddenTabBar on every dock of one
        // panel, the viewer's dock had NoTabBar, and nothing clears either
        // now. Both go from every dock once the layout is loaded, and the
        // arrangement stays. (The viewer's NoUndocking and NoDockingOverMe
        // came from its window class each frame and were never saved.) True
        // when a dock changed.
        bool showHiddenTabBars() {
            const ImGuiDockNodeFlags hidden = ImGuiDockNodeFlags_NoTabBar | ImGuiDockNodeFlags_HiddenTabBar;
            bool changed = false;
            for (ImGuiStoragePair& entry : GImGui->DockContext.Nodes.Data) {
                auto* node = static_cast<ImGuiDockNode*>(entry.val_p);
                if (!node || (node->LocalFlags & hidden) == 0) continue;
                node->SetLocalFlags(node->LocalFlags & ~hidden);
                changed = true;
            }
            return changed;
        }

        // --- the boxes --------------------------------------------------------

        class MessageBox final : public Dialog {
        public:
            MessageBox(std::string title, std::string text, std::vector<std::string> buttons, std::function<void(int)> answer)
                : title_(std::move(title)), text_(std::move(text)), buttons_(std::move(buttons)), answer_(std::move(answer)) {}

            std::string title() const override { return title_; }
            ImVec2 size() const override { return ImVec2(460, 0); }

            void draw(App&) override {
                widgets::textWrapped(text_, 13, theme::kText);
                widgets::vspace(10);
                // buttons flush right, the default (last) one primary
                float total = 0.0f;
                std::vector<float> widths;
                for (std::size_t i = 0; i < buttons_.size(); ++i) {
                    const bool primary = i + 1 == buttons_.size();
                    const float w = std::max(px(84), theme::textSize(buttons_[i], 13, primary ? Weight::ExtraBold : Weight::SemiBold).x + px(28));
                    widths.push_back(w);
                    total += w + (i ? px(8) : 0.0f);
                }
                ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - total));
                for (std::size_t i = 0; i < buttons_.size(); ++i) {
                    if (i) ImGui::SameLine(0.0f, px(8));
                    widgets::ButtonOpts o;
                    o.kind = i + 1 == buttons_.size() ? widgets::ButtonKind::Primary : widgets::ButtonKind::Secondary;
                    o.width = widths[i] / std::max(theme::scale(), 0.01f);
                    o.centered = true;
                    // Enter answers the box on top only: a box under a nested one
                    // would otherwise take it first, as it is drawn first.
                    const bool enter = onTop() && i + 1 == buttons_.size() &&
                                       (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false));
                    if (widgets::button((buttons_[i] + "##" + std::to_string(i)).c_str(), o) || enter) {
                        chosen_ = static_cast<int>(i);
                        close();
                    }
                }
            }

            void closed(App&) override {
                if (answer_) answer_(chosen_);
            }

        private:
            std::string title_, text_;
            std::vector<std::string> buttons_;
            std::function<void(int)> answer_;
            int chosen_ = -1;
        };

        class TextPrompt final : public Dialog {
        public:
            TextPrompt(std::string title, std::string label, std::string initial, std::function<void(const std::string&)> accepted,
                       bool password)
                : title_(std::move(title)), label_(std::move(label)), value_(std::move(initial)), accepted_(std::move(accepted)),
                  password_(password) {}

            std::string title() const override { return title_; }
            ImVec2 size() const override { return ImVec2(420, 0); }

            void draw(App&) override {
                widgets::fieldLabel(label_);
                if (first_) ImGui::SetKeyboardFocusHere();
                first_ = false;
                widgets::FieldOpts f;
                f.enterReturnsTrue = true;
                f.password = password_;
                const bool enter = widgets::inputText("##value", &value_, f);
                widgets::vspace(8);
                const float w = px(84);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - 2 * w - px(8));
                widgets::ButtonOpts cancel;
                cancel.width = 84;
                cancel.centered = true;
                if (widgets::button("Cancel", cancel)) close();
                ImGui::SameLine(0.0f, px(8));
                widgets::ButtonOpts ok;
                ok.kind = widgets::ButtonKind::Primary;
                ok.width = 84;
                ok.centered = true;
                if (widgets::button("OK", ok) || enter) {
                    ok_ = true;
                    close();
                }
            }

            void closed(App&) override {
                if (ok_ && accepted_) accepted_(value_);
            }

        private:
            std::string title_, label_, value_;
            std::function<void(const std::string&)> accepted_;
            bool first_ = true;
            bool ok_ = false;
            bool password_ = false;
        };

    } // namespace

    std::string shortcutText(ImGuiKeyChord chord) {
        if (chord == 0) return std::string();
        std::string out;
#ifdef __APPLE__
        if (chord & ImGuiMod_Ctrl) out += "\xE2\x8C\x98";    // Dear ImGui swaps Cmd and Ctrl on macOS
        if (chord & ImGuiMod_Alt) out += "\xE2\x8C\xA5";
        if (chord & ImGuiMod_Shift) out += "\xE2\x87\xA7";
#else
        if (chord & ImGuiMod_Ctrl) out += "Ctrl+";
        if (chord & ImGuiMod_Alt) out += "Alt+";
        if (chord & ImGuiMod_Shift) out += "Shift+";
        if (chord & ImGuiMod_Super) out += "Super+";
#endif
        out += keyName(static_cast<ImGuiKey>(chord & ~ImGuiMod_Mask_));
        return out;
    }

    // -------------------------------------------------------------------------
    // Impl
    // -------------------------------------------------------------------------

    struct Action {
        std::string menu;                         // "File"; "" = a shortcut without a menu entry
        std::string text;                         // "Open dataset…"
        std::vector<ImGuiKeyChord> keys;          // the first one is shown
        std::function<void()> run;
        std::function<bool()> enabled;            // null = always
        std::function<bool()> checked;            // null = not checkable
        std::function<std::string()> label;       // null = text
        std::string tip;
        bool separatorBefore = false;
        bool recentMenu = false;                  // the "Open recent" submenu
        bool frozenByRun = false;                 // refused while a run or a load is in progress
    };

    struct App::Impl {
        App& self;
        explicit Impl(App& app) : self(app) {}

        GLFWwindow* window = nullptr;
        bool unattended = false;
        bool visible = true;
        bool closing = false;                 // the loop ends
        bool closeAsked = false;              // the user asked; the questions are on screen
        int exitCode = 0;
        int redrawFrames = 3;
        int frameDepth = 0;
        std::uint64_t frames = 0;
        bool glfwReady = false, imguiReady = false;
        bool viewports = false;               // the user wants floating windows (ui/floatingWindows)
        bool viewportsHeldBack = false;       // ... but the monitors' scales differ (updateViewports)
        // Window coordinates per design pixel where the window is now (monitorScale):
        // the design pixels' factor, before the user's zoom.
        float contentScale = 1.0f;
        // Inside the frame loop's glfwPollEvents: the only place a refresh
        // callback may draw (not a native dialog's modal loop, not a frame).
        bool inPoll = false;

        std::unique_ptr<Viewer> viewer;
        std::unique_ptr<OpsPanel> ops;
        std::unique_ptr<ParamsPanel> params;
        std::unique_ptr<DiagnosticsPanel> diagnostics;
        std::unique_ptr<AssistantPanel> assistant;
        std::unique_ptr<LogPanel> log;
        std::unique_ptr<HelpWindow> help;
        std::unique_ptr<ClusterLink> cluster;
        std::shared_ptr<PluginManager> plugins;   // created on first use

        std::vector<std::shared_ptr<Dialog>> dialogs;
        std::vector<std::function<void()>> deferred;
        std::vector<Action> actions;

        // dock windows
        bool showOps = true, showParams = true, showDiag = true, showLog = true, showAssistant = false;
        bool focusOps = false, focusDiag = false, focusLog = false, focusAssistant = false;
        int diagFrontFrames = 0;
        // A non-modal dialog (the plugin manager's editor) had the keyboard in
        // the frame before: the application's shortcuts leave its keys alone,
        // or Backspace would remove a step and Ctrl+S save the pipeline.
        bool dialogHadFocus = false, dialogHasFocus = false;
        std::vector<ImGuiKeyChord> claimedKeys, claimingKeys;   // App::claimKey: last frame's, this frame's               // after the layout is built: Diagnostics, not the log, in front
        bool layoutBuilt = false;
        bool rebuildLayout = false;
        bool assistantPlaced = false;          // the assistant dock was given its place once
        // The layout was just rebuilt: the assistant is placed again even though
        // imgui.ini knows it (a saved arrangement is only kept at start-up).
        bool forcePlaceAssistant = false;
        ImGuiID dockspace = 0;
        ImGuiID assistantDockTarget = 0;       // the node right of Parameters
        // The diagnostics' maximise: they cover the viewer (the design's
        // "maximise over viewer"), where every other panel covers the whole
        // main window (Place::maximized).
        bool diagMaximized = false;
        // Where each panel is, as its window was last drawn, and where the
        // controls on its tab bar send it (indexed by Panel). Docked means
        // docked in the main dockspace: a group floating on its own (on
        // another monitor, say) is floating. A maximised panel is drawn by a
        // stand-in window over the main window; its own window is not drawn
        // meanwhile and keeps its place, its dock or where it floats, to go
        // back to (and that place is what the layout saves).
        struct Place {
            const char* window;                // its window's name
            ImGuiDir home;                     // the side it docks to when its dock is gone (None: the central node; dockTarget)
            float across;                      // its design size across that side
            ImGuiID dock = 0;                  // the dock of the main window it was last in, to put it back
            bool floating = false;             // not in the main dockspace
            bool maximized = false;            // over the main window, in its stand-in (never the diagnostics)
            Move request = Move::None;         // what its window's next Begin does
            ImVec2 min{0, 0}, max{0, 0};       // its window, as last drawn in front
        };
        std::array<Place, kPanelCount> places{{{kViewerWindow, ImGuiDir_None, 0.0f},
                                               {kOpsWindow, ImGuiDir_Left, theme::kOpsDockW},
                                               {kParamsWindow, ImGuiDir_Right, theme::kParamsDockW},
                                               {kAssistantWindow, ImGuiDir_Right, theme::kAssistantW},
                                               {kLogWindow, ImGuiDir_Down, theme::kDiagnosticsH},
                                               {kDiagWindow, ImGuiDir_Down, theme::kDiagnosticsH}}};
        Place& placeOf(Panel p) { return places[static_cast<std::size_t>(p)]; }
        // Dear ImGui gives a new dock the lowest id no dock has, so the id of a
        // dock that is gone soon names another one, where Dock would put the
        // panel as a tab. A panel's dock is forgotten once it is gone, before
        // new docks are made: before the docks asked for are found and after
        // the dockspace is laid out, on every frame.
        void forgetLostDocks() {
            for (Place& pl : places)
                if (pl.dock && !ImGui::DockBuilderGetNode(pl.dock)) pl.dock = 0;
        }
        // One panel at a time is maximised: every maximised panel but `keep`
        // goes back to its place.
        void restoreMaximized(Panel keep) {
            for (std::size_t i = 0; i < kPanelCount; ++i) {
                if (static_cast<Panel>(i) == keep || !places[i].maximized) continue;
                places[i].maximized = false;
                self.requestRedraw();
            }
        }
        // The panel's Window-menu flag; the viewer has none, it is always shown.
        bool* shownFlag(Panel p) {
            switch (p) {
                case Panel::Ops: return &showOps;
                case Panel::Params: return &showParams;
                case Panel::Assistant: return &showAssistant;
                case Panel::Log: return &showLog;
                case Panel::Diag: return &showDiag;
                case Panel::Viewer: break;
            }
            return nullptr;
        }
        // The panel's collapsed state as the window's size last followed it,
        // the height it was collapsed to, the height to give back on
        // expanding, and the height of what sits above the panel's header in
        // its window (the tab bar of its dock, docked or floating).
        // diagFitPlace is where the window was then: its dock node, or 0 on
        // its own.
        bool diagCollapsed = false;
        float diagCollapsedH = 0.0f, diagExpandedH = 0.0f, diagInset = 0.0f;
        ImGuiID diagFitPlace = 0;
        ImVec2 viewerMin{0, 0}, viewerMax{0, 0};
        // The viewer was drawn docked in the main window with its tab in
        // front (this frame once it is drawn, the last one between frames):
        // viewerMin / viewerMax are its room below the tab bar now, which
        // maximised diagnostics cover. viewerDrawn: the viewer drew itself,
        // neither behind a tab nor under the cover; a hidden one does not
        // keep the frames coming for its playback.
        bool viewerInFront = false, viewerDrawn = false;
        bool openAddMenu = false;

        // status bar
        std::string logLine;
        Clock::time_point logLineAt{};
        bool progressActive = false;
        Clock::time_point progressStart{};
        std::string progressMessage;
        std::string lastDir;
        std::string windowTitle;
        // The history's revision when the pipeline was last saved or loaded:
        // any other one is unsaved work (label edits are never saved by
        // File > Save; they leave through an export).
        std::uint64_t savedRevision = 0;

        std::string pendingScreenshot;
        bool screenshotOk = false;
        std::vector<std::string> gpuItems;
        std::vector<int> gpuIds;

        Workbench& wb() { return self.wb(); }
        Bridge& bridge() { return self.bridge(); }

        // --- building -----------------------------------------------------------
        Action& add(const std::string& menu, const std::string& text, std::vector<ImGuiKeyChord> keys, std::function<void()> run,
                    const std::string& tip = {}) {
            Action a;
            a.menu = menu;
            a.text = text;
            a.keys = std::move(keys);
            a.run = std::move(run);
            a.tip = tip;
            a.separatorBefore = pendingSeparator;
            pendingSeparator = false;
            actions.push_back(std::move(a));
            return actions.back();
        }
        bool pendingSeparator = false;
        void separator() { pendingSeparator = true; }

        bool canEditNow() { return wb().canEdit() && !bridge().taskRunning(); }
        bool busy() { return bridge().running() || bridge().taskRunning(); }
        bool stepOk() {
            const int i = wb().selectedIndex();
            return i >= 0 && i < wb().pipeline().size();
        }
        bool movable() { return stepOk() && wb().selectedIndex() > 0; }

        void buildActions();
        void buildDefaultLayout(ImGuiID dockspaceId, ImVec2 size);
        ImGuiID dockTarget(Panel p);
        void followDiagnosticsCollapse();
        bool fitDiagnosticsToCollapse(bool collapsed);

        // --- drawing ---------------------------------------------------------------
        void drawFrame(bool fromRefresh);
        void abandonFrame(const std::string& what);
        void drawTitleBar();
        void drawMenu(const std::string& name);
        bool drawMenuItem(Action& a);
        void drawStatusBar();
        void drawDockWindows();
        bool beginPanel(Panel p, bool* open, ImGuiID dockTo);
        bool beginStandIn(Panel p, bool* open, ImVec2 min, ImVec2 max);
        void endStandIn(Panel p, ImVec2 min, ImVec2 max);
        void drawWindowControls(Panel p, ImGuiWindow* panelWindow, bool standIn);
        void dockPanel(Panel p);
        void floatPanel(Panel p);
        void toggleMaximized(Panel p);
        void drawDialogs(std::size_t index);
        void drawDialogBody(Dialog& d);
        void handleShortcuts();
        void refreshTitle();
        void updateProgress(double& fraction, std::string& message, bool& active);
        bool unsavedWork() { return wb().history().revision() != savedRevision; }
        void finishClose();
        void applyScale();
        void rescale();
        void updateViewports();
        void captureScreenshot();
    };

    void App::Impl::buildActions() {
        actions.clear();
        auto edit = [this] { return canEditNow(); };
        auto viewFlag = [this](bool ViewState::* member) {
            return [this, member] {
                ViewState s = wb().viewState();
                s.*member = !(s.*member);
                wb().setViewState(s);
            };
        };

        // File
        add("File", "Open dataset\xE2\x80\xA6", {ImGuiMod_Ctrl | ImGuiKey_O}, [this] { self.openDatasetDialog(); });
        add("File", "Open folder as dataset\xE2\x80\xA6", {ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_O}, [this] { self.openFolderDataset(); });
        {
            // a dataset that stays on the cluster: its folders through the SSH session
            const auto openFromCluster = [this] {
                auto open = [this](const std::string& path) { self.openDatasetPath(path); };
                self.showDialog(makeClusterBrowser(self, {}, false, open));
            };
            add("File", "Open from cluster\xE2\x80\xA6", {}, openFromCluster, "A dataset on the cluster, read there by the worker");
        }
        add("File", "Open recent", {}, nullptr).recentMenu = true;
        {
            Action& a = add("File", "Close dataset", {ImGuiMod_Ctrl | ImGuiKey_W}, [this] {
                if (bridge().taskRunning()) bridge().cancelTask();
                wb().closeDataset();
            });
            a.enabled = [this] { return (canEditNow() && wb().hasDataset()) || bridge().taskRunning(); };
        }
        separator();
        add("File", "Save pipeline", {ImGuiMod_Ctrl | ImGuiKey_S}, [this] { self.savePipelineTo(wb().pipelinePath()); });
        add("File", "Save pipeline as\xE2\x80\xA6", {ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S}, [this] { self.savePipelineTo(std::string()); });
        {
            // It replaces the pipeline: refused while a run or a load holds it,
            // or a finishing load would install its dataset over the new pipeline.
            Action& a = add("File", "Load pipeline preset\xE2\x80\xA6", {}, [this] { self.loadPipeline(); });
            a.enabled = edit;
            a.frozenByRun = true;
        }
        separator();
        add("File", "Export result\xE2\x80\xA6", {ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_E}, [this] { self.exportResultDialog(); }).enabled =
            [this] { return wb().hasDataset() && !busy(); };
        add("File", "Export training data\xE2\x80\xA6", {}, [this] { self.exportTrainingDialog(); }, "Write the labels of a step as instance masks, bounding boxes and a semantic mask into a dataset folder").enabled = [this] { return wb().hasDataset() && !busy(); };
        add("File", "Export pipeline as Python script\xE2\x80\xA6", {}, [this] { self.exportPythonScript(); });
        separator();
        {
            Action& a = add("File", "Record session\xE2\x80\xA6", {}, [this] { self.toggleRecording(); }, "Write what you do to a JSON-lines file: steps, parameter changes, run results and label corrections");
            a.label = [this] {
                return wb().recording() ? format("Stop recording (%llu events)", static_cast<unsigned long long>(wb().recordedLines()))
                                        : std::string("Record session\xE2\x80\xA6");
            };
        }
        add("File", "Export figure (current view)\xE2\x80\xA6", {ImGuiMod_Ctrl | ImGuiMod_Alt | ImGuiKey_E}, [this] { self.exportFigureImage(); });
        separator();
        add("File", "Preferences\xE2\x80\xA6", {ImGuiMod_Ctrl | ImGuiKey_Comma}, [this] { self.preferences(); });
        separator();
        add("File", "Quit", {ImGuiMod_Ctrl | ImGuiKey_Q}, [this] { self.requestClose(); });

        // Edit
        {
            Action& a = add("Edit", "Undo", {ImGuiMod_Ctrl | ImGuiKey_Z}, [this] { wb().undo(); });
            a.enabled = [this] { return canEditNow() && wb().history().canUndo(); };
            a.label = [this] { return wb().history().canUndo() ? "Undo " + wb().history().undoLabel() : std::string("Undo"); };
            a.frozenByRun = true;
        }
        {
            Action& a = add("Edit", "Redo", {ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Z, ImGuiMod_Ctrl | ImGuiKey_Y}, [this] { wb().redo(); });
            a.enabled = [this] { return canEditNow() && wb().history().canRedo(); };
            a.label = [this] { return wb().history().canRedo() ? "Redo " + wb().history().redoLabel() : std::string("Redo"); };
            a.frozenByRun = true;
        }
        separator();
        {
            Action& a = add("Edit", "Duplicate step", {keys::duplicateStep}, [this] { wb().duplicateStep(wb().selectedIndex()); });
            a.enabled = [this] { return canEditNow() && movable(); };
            a.frozenByRun = true;
        }
        {
            Action& a = add("Edit", "Remove step", {keys::removeStep}, [this] { self.removeStepAt(wb().selectedIndex()); });
            a.enabled = [this] { return canEditNow() && movable(); };
            a.frozenByRun = true;
        }
        {
            Action& a = add("Edit", "Enable / skip step", {keys::enableStep}, [this] {
                const int i = wb().selectedIndex();
                if (i > 0 && i < wb().pipeline().size()) wb().setStepEnabled(i, !wb().pipeline().at(i).enabled);
            });
            a.enabled = [this] { return canEditNow() && movable(); };
            a.frozenByRun = true;
        }
        separator();
        {
            Action& a = add("Edit", "Move step up", {keys::moveUp}, [this] { wb().moveStep(wb().selectedIndex(), -1); });
            a.enabled = [this] { return canEditNow() && movable() && wb().selectedIndex() > 1; };
            a.frozenByRun = true;
        }
        {
            Action& a = add("Edit", "Move step down", {keys::moveDown}, [this] { wb().moveStep(wb().selectedIndex(), +1); });
            a.enabled = [this] { return canEditNow() && movable() && wb().selectedIndex() < wb().pipeline().size() - 1; };
            a.frozenByRun = true;
        }
        separator();
        add("Edit", "Copy parameters", {ImGuiMod_Ctrl | ImGuiKey_C}, [this] { wb().copyParameters(wb().selectedIndex()); });
        {
            Action& a = add("Edit", "Paste parameters", {ImGuiMod_Ctrl | ImGuiKey_V}, [this] {
                if (!wb().pasteParameters(wb().selectedIndex()))
                    wb().logLine("Paste parameters: the copied parameters belong to another operation kind.");
            });
            a.enabled = [this] { return canEditNow() && wb().hasCopiedParameters() && stepOk(); };
            a.frozenByRun = true;
        }

        // View
        add("View", "Ortho views", {ImGuiKey_1}, [this] { wb().setViewMode(ViewMode::Ortho); }).checked =
            [this] { return wb().viewState().mode == ViewMode::Ortho; };
        add("View", "3D volume", {ImGuiKey_2}, [this] { wb().setViewMode(ViewMode::Volume); }).checked =
            [this] { return wb().viewState().mode == ViewMode::Volume; };
        add("View", "Compare raw vs. step", {ImGuiKey_3}, [this] { wb().setViewMode(ViewMode::Compare); }).checked =
            [this] { return wb().viewState().mode == ViewMode::Compare; };
        separator();
        add("View", "Crosshair", {ImGuiKey_H}, [this] { wb().toggleCrosshair(); }).checked = [this] { return wb().viewState().crosshair; };
        add("View", "Labels overlay", {ImGuiKey_L}, [this] { wb().toggleLabels(); }).checked = [this] { return wb().viewState().labels; };
        add("View", "Scale bar", {}, viewFlag(&ViewState::scaleBar)).checked = [this] { return wb().viewState().scaleBar; };
        add("View", "Physical z scaling", {}, viewFlag(&ViewState::physicalZ),
            "Scale the XZ / YZ panes by the voxel aspect. Off draws one row per plane, for checking the grid rather than the shape")
            .checked = [this] { return wb().viewState().physicalZ; };
        separator();
        add("View", "Zoom in", {ImGuiKey_Equal, ImGuiMod_Shift | ImGuiKey_Equal, ImGuiKey_KeypadAdd}, [this] { viewer->zoomIn(); });
        add("View", "Zoom out", {ImGuiKey_Minus, ImGuiKey_KeypadSubtract}, [this] { viewer->zoomOut(); });
        add("View", "Fit to window", {ImGuiKey_0}, [this] { viewer->fitToWindow(); });
        separator();
        add("View", "Auto contrast (display)", {ImGuiMod_Shift | ImGuiKey_A}, [this] { viewer->autoContrast(); });
        add("View", "Reset contrast (display)", {ImGuiMod_Shift | ImGuiKey_R}, [this] { viewer->resetContrast(); });
        add("View", "Sync Z / T across viewers", {}, viewFlag(&ViewState::syncZT)).checked = [this] { return wb().viewState().syncZT; };

        // Process
        add("Process", "Add operation\xE2\x80\xA6", {ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_A}, [this] {
            self.showOperations();
            ops->openAddMenu();
        });
        separator();
        add("Process", "Run all enabled", {keys::runAll}, [this] { self.runAll(); }).enabled = [this] { return wb().hasDataset() && !busy(); };
        add("Process", "Run selected step", {keys::runSelected}, [this] { self.runSelectedStep(); }, "Run just this step; its input has to be computed already").enabled = [this] { return wb().hasDataset() && !busy() && stepOk(); };
        add("Process", "Run to selected step", {}, [this] { self.runTo(wb().selectedIndex()); }, "Run every enabled step from the top down to this one").enabled = [this] { return wb().hasDataset() && !busy() && stepOk(); };
        add("Process", "Cancel", {ImGuiKey_Escape}, [this] { self.cancel(); }).enabled = [this] { return busy(); };
        separator();
        {
            Action& a = add("Process", "Clear cache for step", {}, [this] { wb().clearCache(wb().selectedIndex()); });
            a.enabled = [this] { return canEditNow() && stepOk(); };
            a.frozenByRun = true;
        }
        {
            Action& a = add("Process", "Clear all caches", {}, [this] { wb().clearAllCaches(); });
            a.enabled = edit;
            a.frozenByRun = true;
        }
        separator();
        add("Process", "Reload plugins", {}, [this] { self.defer([this] { wb().loadPlugins(true); }); });
        separator();
        {
            Action& a = add("Process", "Backend: CUDA", {}, [this] { wb().setBackend(Backend::Cuda); }, cudaAvailable() ? "" : "No CUDA device available");
            a.checked = [this] { return wb().backend() == Backend::Cuda; };
            a.enabled = [] { return cudaAvailable(); };
        }
        add("Process", "Backend: CPU", {}, [this] { wb().setBackend(Backend::Cpu); }).checked = [this] { return wb().backend() == Backend::Cpu; };
        add("Process", "Backend: HPC (Slurm)", {}, [this] { wb().setBackend(Backend::Hpc); }).checked =
            [this] { return wb().backend() == Backend::Hpc; };
        // scripting (--action): Connect with the stored profile, as the dialog's button does
        add("", "Connect to cluster (stored profile)", {}, [this] { self.cluster().connect(self.cluster().storedProfile()); });
        add("", "Log in to cluster (stored profile)", {}, [this] { self.cluster().logIn(self.cluster().storedProfile()); });
        add("", "Wait for the cluster", {}, [this] { self.waitUntil([this] { return self.cluster().status().state != cluster::State::Connecting; }); });
        add("Process", "Connect to cluster\xE2\x80\xA6", {}, [this] { self.clusterDialog(); }, "One SSH login: the worker job, the HPC backend through it, the cluster's datasets");

        // Segment
        {
            // an edit (it adds or changes a step): refused during a run like the label edits
            Action& a = add("Segment", "Load Torch model\xE2\x80\xA6", {ImGuiMod_Ctrl | ImGuiKey_M}, [this] { self.loadTorchModel(); });
            a.enabled = edit;
            a.frozenByRun = true;
        }
        add("Segment", "Download model\xE2\x80\xA6", {}, [this] { self.modelHub(); });
        add("Segment", "Run segmentation", {}, [this] {
            const int i = self.segmentationStep();
            if (i < 0) wb().logLine("Run segmentation: add a segmentation step first (Process \xE2\x96\xB8 Add operation).");
            else bridge().startRun(i);
        });
        separator();
        add("Segment", "Paint labels", {ImGuiKey_B}, [this] {
            wb().setTool(ViewerTool::Paint);
            wb().setPaintTool(PaintTool::Brush);
            if (!wb().viewState().labels) wb().toggleLabels();
        });
        add("Segment", "Erase", {ImGuiKey_E}, [this] {
            wb().setTool(ViewerTool::Paint);
            wb().setPaintTool(PaintTool::Erase);
        });
        {
            Action& a = add("Segment", "Merge selected labels", {ImGuiMod_Ctrl | ImGuiKey_G}, [this] { self.mergeLabelsDialog(); });
            a.enabled = edit;
            a.frozenByRun = true;
        }
        add("Segment", "Split label", {ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_G}, [this] {
            wb().setTool(ViewerTool::Paint);
            wb().setPaintTool(PaintTool::Split);
            wb().logLine("Split: click two seeds inside the label in the viewer.");
        });
        {
            Action& a = add("Segment", "Delete label", {ImGuiMod_Ctrl | ImGuiKey_Backspace}, [this] {
                const std::uint32_t id = wb().viewState().selectedLabel;
                if (id) wb().deleteLabel(id);
            });
            a.enabled = edit;
            a.frozenByRun = true;
        }
        separator();
        add("Segment", "Only selected label", {ImGuiKey_O}, [this] { wb().toggleSoloLabel(); }, "Draw only the selected label in the slices and in 3D; selecting a label jumps to it").checked = [this] { return wb().viewState().soloLabel; };
        add("Segment", "Next flagged label", {ImGuiKey_RightArrow}, [this] { self.selectFlagged(true); });
        add("Segment", "Previous flagged label", {ImGuiKey_LeftArrow}, [this] { self.selectFlagged(false); });
        separator();
        {
            Action& a = add("Segment", "Accept all reviewed", {}, [this] { wb().acceptAllReviewed(); });
            a.enabled = edit;
            a.frozenByRun = true;
        }
        separator();
        add("Segment", "Export labels\xE2\x80\xA6", {}, [this] { self.exportLabels(); });

        // Window
        add("Window", "Operations", {ImGuiMod_Alt | ImGuiKey_1}, [this] { showOps = !showOps; }).checked = [this] { return showOps; };
        add("Window", "Parameters", {ImGuiMod_Alt | ImGuiKey_2}, [this] { showParams = !showParams; }).checked = [this] { return showParams; };
        add("Window", "Diagnostics", {ImGuiMod_Alt | ImGuiKey_3}, [this] {
            showDiag = !showDiag;
            focusDiag = showDiag;
        }).checked = [this] { return showDiag; };
        add("Window", "Help page", {ImGuiMod_Alt | ImGuiKey_4}, [this] { self.toggleHelp(); }).checked = [this] { return self.helpOpen(); };
        separator();
        // The Float and Maximise controls on the diagnostics' tab bar, from the menu.
        add("Window", "Float diagnostics", {}, [this] { self.floatDiagnostics(); });
        add("Window", "Maximise diagnostics", {}, [this] { self.setDiagnosticsMaximized(!diagMaximized); }, "Let the diagnostics cover the viewer").checked = [this] { return diagMaximized; };
        add("Window", "Reset layout", {}, [this] { self.resetLayout(); });
        add("Window", "Save layout as default", {}, [this] {
            ImGui::SaveIniSettingsToDisk(settings().layoutPath().c_str());
            wb().logLine("Saved this layout: the next session opens with it.");
        });
        separator();
        add("Window", "Assistant", {keys::assistant}, [this] { self.setAssistantVisible(!showAssistant); }).checked =
            [this] { return showAssistant; };
        add("Window", "User operations\xE2\x80\xA6", {ImGuiMod_Alt | ImGuiKey_6}, [this] { self.pluginManager(); });
        add("Window", "Log", {keys::logDock}, [this] {
            showLog = !showLog;
            focusLog = showLog; }, "Everything this session has reported").checked = [this] { return showLog; };

        // Help
        add("Help", "Help for this step", {keys::helpForStep}, [this] { self.showHelpForSelected(); });
        add("Help", "Sirius manual", {}, [this] { help->showManual(); });
        add("Help", "Keyboard shortcuts", {ImGuiMod_Ctrl | ImGuiKey_Slash}, [this] { help->showShortcuts(); });
        separator();
        add("Help", "Operation plugin API", {}, [this] { help->showKind("plugin-api"); });
        add("Help", "Report a problem\xE2\x80\xA6", {}, [] { platform::openUrl(kIssuesUrl); });
        separator();
        add("Help", "About Sirius", {}, [this] { self.about(); });
    }

    // The arrangement of docs/design: Operations on the left, Parameters (and
    // the assistant) on the right, both the full height; Diagnostics and the
    // log share the bottom of what is left, under the viewer in the central
    // node. Where they go from there is the user's: every one is moved by its tab.
    void App::Impl::buildDefaultLayout(ImGuiID id, ImVec2 size) {
        ImGui::DockBuilderRemoveNode(id);
        ImGui::DockBuilderAddNode(id, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(id, size);
        ImGuiID centre = id, left = 0, right = 0, bottom = 0;
        const float w = std::max(size.x, 1.0f), h = std::max(size.y, 1.0f);
        ImGui::DockBuilderSplitNode(centre, ImGuiDir_Left, std::clamp(px(theme::kOpsDockW) / w, 0.05f, 0.4f), &left, &centre);
        const float rest = std::max(w - px(theme::kOpsDockW), 1.0f);
        ImGui::DockBuilderSplitNode(centre, ImGuiDir_Right, std::clamp(px(theme::kParamsDockW) / rest, 0.05f, 0.5f), &right, &centre);
        ImGui::DockBuilderSplitNode(centre, ImGuiDir_Down, std::clamp(px(theme::kDiagnosticsH) / h, 0.05f, 0.6f), &bottom, &centre);
        ImGui::DockBuilderDockWindow(kOpsWindow, left);
        ImGui::DockBuilderDockWindow(kParamsWindow, right);
        ImGui::DockBuilderDockWindow(kDiagWindow, bottom);
        // The log shares the bottom area with the diagnostics as a tab: same
        // place, one click away, and it keeps the viewer its full height.
        ImGui::DockBuilderDockWindow(kLogWindow, bottom);
        ImGui::DockBuilderDockWindow(kViewerWindow, centre);
        ImGui::DockBuilderFinish(id);
        // The docks the panels were last in are gone, and the new ones may
        // have their ids: each panel notes its new dock once it is drawn.
        for (Place& pl : places) pl.dock = 0;
        assistantPlaced = false;
        forcePlaceAssistant = true;
        diagFrontFrames = 3;
        // The new bottom dock has the design's height: a collapsed panel
        // collapses it again, and expanding gives back that height.
        diagCollapsed = false;
        diagExpandedH = 0.0f;
    }

    // Where Dock puts a panel back into the main window: the dock it was last
    // in there, else (for the diagnostics and the log, which share the bottom
    // by design) the other one's. Else its side at its design size, as the
    // default arrangement has it: Operations, Parameters and the assistant
    // as a column the main window's full height at the dockspace's outer
    // edge (as a tab dropped on the outer edge target docks: queued, and
    // done by Dear ImGui at the next frame's start, when 0 is returned); the
    // diagnostics and the log in a new split at the bottom of the central
    // node -- under the viewer while it is there; the node may as well hold
    // another panel, or nothing when the viewer floats, and keeps what it
    // has. The viewer goes into the central node itself. Only docks of the
    // main dockspace count: a group floating on its own is not "the main
    // window". The request is done with, unless 0 is returned without one
    // queued: then there was no place.
    ImGuiID App::Impl::dockTarget(Panel p) {
        auto usable = [this](ImGuiID id) {
            ImGuiDockNode* node = id ? ImGui::DockBuilderGetNode(id) : nullptr;
            return node && node->IsLeafNode() && inDockspace(node, dockspace);
        };
        Place& pl = placeOf(p);
        if (usable(pl.dock)) return pl.dock;
        const Panel partner = p == Panel::Diag ? Panel::Log : Panel::Diag;
        if (p == Panel::Diag || p == Panel::Log) {
            ImGuiWindow* pw = ImGui::FindWindowByName(placeOf(partner).window);
            if (pw && *shownFlag(partner) && usable(pw->DockId)) return pw->DockId;
        }
        if (pl.home == ImGuiDir_Left || pl.home == ImGuiDir_Right) {
            ImGuiDockNode* root = ImGui::DockBuilderGetNode(dockspace);
            ImGuiWindow* w = ImGui::FindWindowByName(pl.window);
            if (root && root->HostWindow && w && w != root->HostWindow) {
                // the ratio is the left (first) node's, as DockBuilderSplitNode hands it on
                const float ratio = std::clamp(px(pl.across) / std::max(root->Size.x, 1.0f), 0.05f, 0.6f);
                ImGui::DockContextQueueDock(GImGui, root->HostWindow, root, w, pl.home, pl.home == ImGuiDir_Left ? ratio : 1.0f - ratio, true);
                pl.request = Move::None;
                return 0;
            }
        }
        ImGuiDockNode* centre = centralNode(ImGui::DockBuilderGetNode(dockspace));
        if (!centre) return 0;
        if (pl.home == ImGuiDir_None) return centre->ID;
        const float extent = pl.home == ImGuiDir_Left || pl.home == ImGuiDir_Right ? centre->Size.x : centre->Size.y;
        ImGuiID side = 0, rest = 0;
        ImGui::DockBuilderSplitNode(centre->ID, pl.home, std::clamp(px(pl.across) / std::max(extent, 1.0f), 0.05f, 0.6f), &side, &rest);
        ImGui::DockBuilderFinish(dockspace);
        return side;
    }

    // Asked before the dockspace is laid out, every frame. The panel collapses
    // its window while the panel is what the window shows: with the log's tab
    // in front of it, the shared dock is the log's, at its full height. A
    // collapsed window is fitted again when what sits above the header
    // changes height (at another scale) and when it moves (floated, docked,
    // dropped into another dock): a tab bar of the same height sits above
    // the header wherever it is, so the header alone does not tell.
    void App::Impl::followDiagnosticsCollapse() {
        if (!diagnostics) return;
        const ImGuiWindow* dw = ImGui::FindWindowByName(kDiagWindow);
        const bool behindTab = dw && dw->DockIsActive && dw->DockNode && dw->DockNode->VisibleWindow && dw->DockNode->VisibleWindow != dw;
        const bool collapse = showDiag && diagnostics->isCollapsed() && !behindTab;
        const float collapsedH = diagInset + theme::snap(px(theme::kDiagnosticsHeaderH));
        // Its dock, or 0 on its own. Not DockIsActive: Dear ImGui clears
        // that for a panel alone in its dock until its next Begin, which for a
        // floating dock is before this.
        const ImGuiID place = dw && dw->DockNode ? dw->DockNode->ID : 0;
        if (collapse != diagCollapsed || (collapse && (std::abs(collapsedH - diagCollapsedH) > 0.5f || place != diagFitPlace)))
            fitDiagnosticsToCollapse(collapse);
    }

    // The diagnostics window follows its panel's collapse: collapsed, it keeps
    // only the header, and the viewer takes the room; expanded, it gets its
    // height back. Docked, the dock node is sized (locked once, as a splitter
    // drag does, so the other side of the split takes the difference);
    // floating in a dock of its own (a floating panel keeps its tab bar), that
    // dock, which is the floating window. False when there is nothing to size
    // (yet): a window between two docks (the frame a layout is rebuilt), one
    // in a split floating group, or a dock side by side with another; the
    // next frame tries again.
    bool App::Impl::fitDiagnosticsToCollapse(bool collapsed) {
        ImGuiWindow* dw = ImGui::FindWindowByName(kDiagWindow);
        if (!dw) return false;
        const float collapsedH = diagInset + theme::snap(px(theme::kDiagnosticsHeaderH));
        auto heightFor = [&](float current) {
            if (!collapsed) return diagExpandedH > 0.0f ? diagExpandedH : px(theme::kDiagnosticsH);
            // the height to give back is the expanded one, not a collapsed one fitted again
            if (!diagCollapsed) diagExpandedH = std::max(current, collapsedH + px(60));
            return collapsedH;
        };
        ImGuiDockNode* node = dw->DockNode;
        if (dw->DockIsActive && node && inDockspace(node, dockspace)) {
            // Only a dock stacked above or below another: in a row, its height is the row's.
            if (!node->ParentNode || node->ParentNode->SplitAxis != ImGuiAxis_Y) return false;
            node->Size.y = node->SizeRef.y = heightFor(node->Size.y);
            node->WantLockSizeOnce = true;
        } else if (node && node->IsRootNode() && node->HostWindow && !inDockspace(node, dockspace)) {
            // Floating in a dock of its own (with the log as a tab, maybe):
            // the dock is the floating window. The window's own size is set
            // too: the layout keeps that one for a panel floating alone.
            const ImVec2 size(node->Size.x, heightFor(node->Size.y));
            if (size.x > 0.0f && size.y > 0.0f) ImGui::DockBuilderSetNodeSize(node->ID, size);
            ImGui::SetWindowSize(kDiagWindow, size, ImGuiCond_Always);
        } else if (!dw->DockIsActive && ((node && !node->HostWindow) || dw->DockId == 0)) {
            // On its own: just floated, before its dock has a window (which
            // then takes this window's size). A dock id without a node is a
            // window about to be docked (or hidden from its dock): its node is
            // sized once it is in.
            ImGui::SetWindowSize(kDiagWindow, ImVec2(dw->Size.x, heightFor(dw->Size.y)), ImGuiCond_Always);
        } else {
            return false;
        }
        // The collapsed panel left the dock it had collapsed: what stays there
        // (the log) gets the height back.
        const ImGuiID place = node ? node->ID : 0;
        if (diagCollapsed && diagFitPlace != 0 && diagFitPlace != place) {
            ImGuiDockNode* left = ImGui::DockBuilderGetNode(diagFitPlace);
            if (left && left->IsLeafNode() && inDockspace(left, dockspace) && left->ParentNode && left->ParentNode->SplitAxis == ImGuiAxis_Y) {
                left->Size.y = left->SizeRef.y = diagExpandedH > 0.0f ? diagExpandedH : px(theme::kDiagnosticsH);
                left->WantLockSizeOnce = true;
            }
        }
        diagCollapsed = collapsed;
        diagCollapsedH = collapsedH;
        diagFitPlace = place;
        return true;
    }

    void App::Impl::applyScale() {
        float user = static_cast<float>(settings().getDouble("ui/scale", 1.0));
        const std::string env = platform::environment("SIRIUS_UI_SCALE");
        if (!env.empty()) {
            try {
                user = std::stof(env);
            } catch (const std::exception&) {
            }
        }
        if (!(user > 0.0f)) user = 1.0f;
        theme::setScale(contentScale * user);
        theme::applyTheme();
    }

    // The window moved to a monitor of another scale, or its monitor's scale
    // changed. GLFW keeps the client area's pixel size (GLFW_SCALE_TO_MONITOR is
    // off: init() sizes the window itself), so the window and the docks around
    // the viewer are resized by the ratio of the scales: they keep their design
    // size on the new monitor, and the window fits its work area.
    void App::Impl::rescale() {
        const float next = windowScale(window);
        const float ratio = next / contentScale;
        if (std::abs(ratio - 1.0f) > 0.01f) {
            if (!glfwGetWindowAttrib(window, GLFW_MAXIMIZED) && !glfwGetWindowAttrib(window, GLFW_ICONIFIED) && !glfwGetWindowMonitor(window)) {
                int x = 0, y = 0, w = 0, h = 0;
                glfwGetWindowPos(window, &x, &y);
                glfwGetWindowSize(window, &w, &h);
                const int cx = x + w / 2, cy = y + h / 2;
                w = static_cast<int>(std::lround(static_cast<float>(w) * ratio));
                h = static_cast<int>(std::lround(static_cast<float>(h) * ratio));
                int count = 0;
                GLFWmonitor** monitors = glfwGetMonitors(&count);
                for (int i = 0; i < count; ++i) {
                    int mx = 0, my = 0, mw = 0, mh = 0;
                    glfwGetMonitorWorkarea(monitors[i], &mx, &my, &mw, &mh);
                    if (mw <= 0 || mh <= 0 || cx < mx || cx >= mx + mw || cy < my || cy >= my + mh) continue;
                    w = std::min(w, mw);
                    h = std::min(h, mh - 40);   // the title bar is outside the client area
                    x = std::clamp(x, mx, std::max(mx, mx + mw - w));
                    y = std::clamp(y, my, std::max(my, my + mh - h));
                    break;
                }
                glfwSetWindowSize(window, std::max(w, 1), std::max(h, 1));
                glfwSetWindowPos(window, x, y);
            }
            if (dockspace) scaleDockTree(ImGui::DockBuilderGetNode(dockspace), ratio);
            diagExpandedH *= ratio;   // display pixels too
        }
        contentScale = next;
        applyScale();
    }

    // Floating windows (secondary viewports) are drawn at the main window's
    // scale: on a monitor of another scale they would come out at half or twice
    // the size of the rest of the desktop. With monitors of different scales the
    // panels stay inside the main window. Asked before every frame (as the
    // backend reads the monitors), so a monitor plugged in later counts too;
    // Dear ImGui moves the windows when the flag changes.
    void App::Impl::updateViewports() {
        const bool held = viewports && !monitorsShareScale();
        ImGuiIO& io = ImGui::GetIO();
        if (viewports && !held) io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        else io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;
        if (held != viewportsHeldBack)
            wb().logLine(held ? std::string("Floating windows stay inside the main window: the monitors have different display scales.")
                              : std::string("Floating windows can leave the main window again: the monitors share one display scale."));
        viewportsHeldBack = held;
    }

    void App::Impl::refreshTitle() {
        const Workbench& w = wb();
        std::string title = "SIRIUS";
        if (w.hasDataset()) title += " \xE2\x80\x94 " + w.dataset().name;
        if (!w.pipelinePath().empty()) title += " \xC2\xB7 " + fileName(w.pipelinePath());
        if (title != windowTitle && window) {
            windowTitle = title;
            glfwSetWindowTitle(window, title.c_str());
        }
    }

    // --- menus ---------------------------------------------------------------

    bool App::Impl::drawMenuItem(Action& a) {
        const bool enabled = !a.enabled || a.enabled();
        const bool checked = a.checked && a.checked();
        const std::string label = a.label ? a.label() : a.text;
        const std::string shortcut = a.keys.empty() ? std::string() : shortcutText(a.keys.front());
        const float h = theme::snap(px(27));
        const float left = px(28), right = px(12);
        const float textW = theme::textSize(label, 12).x;
        const float keyW = shortcut.empty() ? 0.0f : theme::textSize(shortcut, 12).x + px(28);
        const float w = std::max(px(240), left + textW + keyW + right);
        ImGui::BeginDisabled(!enabled);
        const ImVec2 min = ImGui::GetCursorScreenPos();
        const bool clicked = ImGui::Selectable(("##" + a.menu + "/" + a.text).c_str(), false, ImGuiSelectableFlags_None, ImVec2(w, h));
        const bool hovered = ImGui::IsItemHovered();
        ImGui::EndDisabled();
        const float width = std::max(w, ImGui::GetItemRectSize().x);
        const ImVec2 max(min.x + width, min.y + h);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImU32 fg = !enabled ? theme::kNeutral500 : (hovered ? theme::kAccentText : theme::kText);
        if (checked) {
            const float s = theme::snap(px(10));
            const ImVec2 c(min.x + px(10), theme::snap(min.y + (h - s) * 0.5f));
            dl->AddRectFilled(c, ImVec2(c.x + s, c.y + s), enabled ? theme::kAccent : theme::kNeutral400);
        }
        widgets::drawTextIn(dl, ImVec2(min.x + left, min.y), ImVec2(max.x, max.y), label, 12, fg, Weight::Regular, 0.0f, 0.5f);
        if (!shortcut.empty())
            widgets::drawTextIn(dl, ImVec2(min.x, min.y), ImVec2(max.x - right, max.y), shortcut, 12,
                                enabled ? theme::kNeutral600 : theme::kNeutral500, Weight::Regular, 1.0f, 0.5f);
        std::string tip = a.tip;
        if (!enabled && a.frozenByRun && busy()) tip = "Not while a run or load is in progress \xE2\x80\x94 cancel it (Esc) or wait";
        if (!tip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) widgets::tooltip(tip);
        return clicked && enabled;
    }

    void App::Impl::drawMenu(const std::string& name) {
        // the menu's title: ink fill and paper text while it is hovered or open
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const ImGuiStyle& style = ImGui::GetStyle();
        const float spacing = std::trunc(style.ItemSpacing.x * 0.5f);
        ImGui::PushStyleColor(ImGuiCol_Text, theme::kTransparent);
        ImGui::PushStyleColor(ImGuiCol_Header, theme::kText);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, theme::kText);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, theme::kText);
        ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(0, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(style.ItemSpacing.x, 0.0f));
        ImDrawList* barList = ImGui::GetWindowDrawList();
        const bool open = ImGui::BeginMenu(name.c_str());
        // The title's own box as Dear ImGui laid it out: the hover test, the
        // ink fill and the paper text all follow it, so they never disagree.
        const ImVec2 hitMin = ImGui::GetItemRectMin(), hitMax = ImGui::GetItemRectMax();
        const bool hot = ImGui::IsMouseHoveringRect(hitMin, hitMax) && ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup |
                                                                                              ImGuiHoveredFlags_ChildWindows);
        ImGui::PopStyleVar(1);   // the item spacing was for the title only
        ImGui::PopStyleColor(4);
        ImGui::PushStyleColor(ImGuiCol_Text, theme::kText);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, theme::kNeutral200);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, theme::kNeutral300);
        if (open || hot) barList->AddRectFilled(hitMin, hitMax, theme::kText);
        barList->AddText(ImVec2(pos.x + spacing, pos.y + style.FramePadding.y), open || hot ? theme::kBg : theme::kText, name.c_str());
        if (open) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
            for (Action& a : actions) {
                if (a.menu != name) continue;
                if (a.separatorBefore) {
                    ImGui::Dummy(px(0, 4));
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + ImGui::GetContentRegionAvail().x, p.y + theme::crispPen(1)),
                                                              theme::kDivider);
                    ImGui::Dummy(ImVec2(0, theme::crispPen(1) + px(4)));
                }
                if (a.recentMenu) {
                    // Dear ImGui's submenu (it opens on hover and stays open while
                    // the mouse travels into it), drawn as the other rows are: its
                    // own label is transparent and set in a font as tall as a row,
                    // so the item is a row high; the label and the arrow are drawn
                    // here at the rows' inset. The font and colour are popped only
                    // after the submenu closed: popped inside it, they would leave
                    // the stack of the window they were pushed in.
                    const float h = theme::snap(px(27));
                    const ImVec2 rowMin = ImGui::GetCursorScreenPos();
                    const ImVec2 rowMax(rowMin.x + std::max(ImGui::GetContentRegionAvail().x, px(240)), rowMin.y + h);
                    ImDrawList* rowList = ImGui::GetWindowDrawList();
                    const bool rowHovered = ImGui::IsMouseHoveringRect(rowMin, rowMax) && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
                    ImGui::PushStyleColor(ImGuiCol_Text, theme::kTransparent);
                    ImGui::PushFont(nullptr, h / std::max(theme::scale(), 0.01f));
                    const bool sub = ImGui::BeginMenu(a.text.c_str());
                    const ImU32 fg = sub || rowHovered ? theme::kAccentText : theme::kText;
                    widgets::drawTextIn(rowList, ImVec2(rowMin.x + px(28), rowMin.y), rowMax, a.text, 12, fg, Weight::Regular, 0.0f, 0.5f);
                    {
                        const float s = px(4), cx = rowMax.x - px(14), cy = (rowMin.y + rowMax.y) * 0.5f;
                        rowList->AddTriangleFilled(ImVec2(cx - s, cy - s), ImVec2(cx - s, cy + s), ImVec2(cx + s * 0.5f, cy), fg);
                    }
                    if (sub) {
                        // the submenu's own rows: the menu's font and ink again
                        ImGui::PushFont(nullptr, 12);
                        ImGui::PushStyleColor(ImGuiCol_Text, theme::kText);
                        const std::vector<std::string> recent = App::recentFiles();
                        if (recent.empty()) {
                            Action none;
                            none.menu = "recent";
                            none.text = "No recent datasets";
                            none.enabled = [] { return false; };
                            drawMenuItem(none);
                        }
                        for (const std::string& path : recent) {
                            Action item;
                            item.menu = "recent";
                            item.text = path;
                            item.label = [path] { return fileName(path); };
                            item.tip = path;
                            if (drawMenuItem(item)) self.defer([this, path] { self.openDatasetPath(path); });
                        }
                        if (!recent.empty()) {
                            ImGui::Dummy(px(0, 4));
                            Action clear;
                            clear.menu = "recent";
                            clear.text = "Clear list";
                            if (drawMenuItem(clear)) settings().remove("recent/datasets");
                        }
                        ImGui::PopStyleColor();
                        ImGui::PopFont();
                        ImGui::EndMenu();
                    }
                    ImGui::PopFont();
                    ImGui::PopStyleColor();
                    continue;
                }
                if (drawMenuItem(a) && a.run) {
                    // outside the menu's popup, between two frames: the action
                    // may open a native dialog or a dialog of ours
                    self.defer(a.run);
                }
            }
            ImGui::PopStyleVar();
            ImGui::EndMenu();
        }
        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(1);
    }

    void App::Impl::drawTitleBar() {
        const float barH = theme::snap(px(theme::kTitleBarH));
        const float fontH = ImGui::GetFontSize();
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(px(9), std::floor((barH - fontH) * 0.5f)));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(9), 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_MenuBarBg, theme::kBg);
        const bool open = ImGui::BeginMainMenuBar();
        ImGui::PopStyleColor();
        if (open) {
            ImGuiWindow* bar = ImGui::GetCurrentWindow();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 origin = ImGui::GetWindowPos();
            const float width = ImGui::GetWindowSize().x;
            const float height = bar->MenuBarHeight;
            // the 2 px rule under the bar
            dl->AddRectFilled(ImVec2(origin.x, origin.y + height - theme::crispPen(theme::kRule)), ImVec2(origin.x + width, origin.y + height),
                              theme::kDivider);

            // brand: 12 x 12 accent square + "SIRIUS" 15 px / 800
            const float s = theme::snap(px(12));
            const ImVec2 chip(origin.x + px(14), theme::snap(origin.y + (height - s) * 0.5f));
            dl->AddRectFilled(chip, ImVec2(chip.x + s, chip.y + s), theme::kAccent);
            const ImVec2 brand = theme::textSize("SIRIUS", theme::kBrandPx, Weight::ExtraBold);
            widgets::drawText(dl, ImVec2(chip.x + s + px(8), origin.y + (height - brand.y) * 0.5f), "SIRIUS", theme::kBrandPx, theme::kText,
                              Weight::ExtraBold);
            ImGui::SetCursorPosX(px(14) + s + px(8) + brand.x + px(14));

            for (const char* menu : {"File", "Edit", "View", "Process", "Segment", "Window", "Help"}) drawMenu(menu);

            // right: dataset · GPU · Assistant
            const Workbench& w = wb();
            const std::string name = w.hasDataset() ? w.dataset().name : std::string("no dataset");
            gpuItems.clear();
            gpuIds.clear();
            const int n = cudaDeviceCount();
            for (int i = 0; i < n; ++i) {
                gpuItems.push_back(deviceLabel(i));
                gpuIds.push_back(i);
            }
            if (n > 1) {
                gpuItems.push_back(format("All %d GPUs", n));
                gpuIds.push_back(Workbench::kAllCudaDevices);
            }
            if (n == 0) {
                gpuItems.push_back("CPU only");
                gpuIds.push_back(0);
            }
            float comboW = px(150);
            for (const std::string& item : gpuItems) comboW = std::max(comboW, theme::textSize(item, 13).x + px(44));
            comboW = std::min(comboW, px(300));
            const float buttonW = theme::textSize("Assistant", 12, Weight::ExtraBold).x + px(40);
            const float nameRoom = std::max(px(40), width - ImGui::GetCursorPosX() - comboW - buttonW - px(14) - 3 * px(18));
            const std::string shown = widgets::elideText(name, std::min(nameRoom, px(320)), 12);
            const float nameW = theme::textSize(shown, 12).x;
            float x = width - px(14) - buttonW - px(18) - comboW - px(18) - nameW;
            if (x > ImGui::GetCursorPosX()) {
                widgets::drawTextIn(dl, ImVec2(origin.x + x, origin.y), ImVec2(origin.x + x + nameW, origin.y + height), shown, 12,
                                    theme::kNeutral600, Weight::Regular, 0.0f, 0.5f);
                x += nameW + px(18);
                ImGui::SetCursorPos(ImVec2(x, std::floor((height - px(26)) * 0.5f)));
                int current = 0;
                for (std::size_t i = 0; i < gpuIds.size(); ++i)
                    if (gpuIds[i] == w.cudaDevice()) current = static_cast<int>(i);
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, px(8, 8));
                widgets::FieldOpts f;
                f.width = comboW / std::max(theme::scale(), 0.01f);
                f.height = 26;
                f.enabled = n > 0;
                if (widgets::combo("##gpu", &current, gpuItems, f) && cudaAvailable()) {
                    // choosing a GPU is choosing to run on it; say so when that
                    // changes the backend
                    if (wb().backend() != Backend::Cuda)
                        wb().logLine("Backend: CUDA (a GPU was chosen; Process \xE2\x96\xB8 Backend switches back).");
                    wb().setBackend(Backend::Cuda);
                    wb().setCudaDevice(gpuIds[static_cast<std::size_t>(current)]);
                }
                ImGui::PopStyleVar();
                x += comboW + px(18);

                // "Assistant" toggle: 26 px, 1.5 px border, accent fill when open
                const float bh = theme::snap(px(26));
                const ImVec2 bmin(origin.x + x, theme::snap(origin.y + (height - bh) * 0.5f)), bmax(bmin.x + buttonW, bmin.y + bh);
                ImGui::SetCursorScreenPos(bmin);
                if (ImGui::InvisibleButton("##assistant", ImVec2(buttonW, bh))) self.setAssistantVisible(!showAssistant);
                const bool hovered = ImGui::IsItemHovered();
                if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                widgets::tooltip(widgets::withShortcut("Assistant", shortcutText(keys::assistant)));
                if (showAssistant) dl->AddRectFilled(bmin, bmax, theme::kAccent);
                widgets::crispRect(dl, bmin, bmax, showAssistant || hovered ? theme::kAccent : theme::kText, theme::kBorder);
                const ImU32 fg = showAssistant ? theme::kBg : theme::kText;
                drawIcon(dl, ImVec2(bmin.x + px(10), bmin.y + (bh - px(13)) * 0.5f), ImVec2(bmin.x + px(23), bmin.y + (bh + px(13)) * 0.5f),
                         Icon::Sparkle, fg);
                widgets::drawTextIn(dl, ImVec2(bmin.x + px(27), bmin.y), ImVec2(bmax.x - px(10), bmax.y), "Assistant", 12, fg,
                                    Weight::ExtraBold, 0.0f, 0.5f);
            }
            ImGui::EndMainMenuBar();
        }
        ImGui::PopStyleVar(3);
    }

    // --- status bar -------------------------------------------------------------

    void App::Impl::updateProgress(double& fraction, std::string& message, bool& active) {
        active = false;
        fraction = 0.0;
        message.clear();
        if (bridge().running()) {
            active = true;
            fraction = bridge().runFraction();
            message = bridge().runMessage();
            const int step = bridge().runStep();
            if (step >= 0 && step < wb().pipeline().size()) {
                const std::string name = wb().pipeline().at(step).name;
                message = message.empty() ? name : name + " \xC2\xB7 " + message;
            }
        } else if (bridge().taskRunning()) {
            active = true;
            fraction = bridge().taskFraction();
            message = bridge().taskMessage();
            if (message.empty()) message = bridge().taskLabel();
        } else if (viewer && viewer->loading()) {
            active = true;
            fraction = viewer->loadFraction();
            message = viewer->loadMessage();
            if (message.empty()) message = "Loading TIFF";
        }
        if (active && !progressActive) {
            progressStart = Clock::now();
            progressMessage.clear();
        }
        progressActive = active;
        if (active && !message.empty()) progressMessage = message;
        message = progressMessage;
    }

    void App::Impl::drawStatusBar() {
        const float h = theme::snap(px(theme::kStatusBarH));
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(1, 1));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBg);
        const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNav |
                                       ImGuiWindowFlags_NoBringToFrontOnFocus;
        if (ImGui::BeginViewportSideBar("##statusbar", vp, ImGuiDir_Down, h, flags)) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 min = ImGui::GetWindowPos();
            const ImVec2 max(min.x + ImGui::GetWindowSize().x, min.y + ImGui::GetWindowSize().y);
            dl->AddRectFilled(min, ImVec2(max.x, min.y + theme::crispPen(theme::kRule)), theme::kDivider);
            const float top = min.y + theme::crispPen(theme::kRule);
            const Workbench& w = wb();

            // right block first: it keeps its place, the left one gives way
            const Pipeline& p = w.pipeline();
            const bool lazy = !w.output(0) || !w.output(0)->array;
            const std::string right = format("%d of %d steps enabled \xC2\xB7 %s \xC2\xB7 %s cached", p.enabledCount(), p.size(),
                                             lazy ? "lazy" : "in memory", bytesText(w.cachedBytes()).c_str());
            const float rightW = theme::textSize(right, 11).x;
            widgets::drawTextIn(dl, ImVec2(max.x - px(14) - rightW, top), ImVec2(max.x - px(14), max.y), right, 11, theme::kNeutral600,
                                Weight::Regular, 0.0f, 0.5f);
            float limit = max.x - px(14) - rightW - px(24);
            // The cluster session, when there is one: connected (green), connecting, or
            // disconnected and why (red); a click opens Connect to cluster.
            ImU32 hpcColor = theme::kNeutral600;
            if (const std::string hpc = cluster ? cluster->indicator(hpcColor) : std::string(); !hpc.empty()) {
                const std::string shown = widgets::elideText(hpc, px(360), 11);
                const float wd = theme::textSize(shown, 11).x;
                const float x0 = limit - wd;
                ImGui::SetCursorScreenPos(ImVec2(x0, top));
                if (ImGui::InvisibleButton("##hpcStatus", ImVec2(std::max(wd, 1.0f), max.y - top))) self.clusterDialog();
                if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                widgets::tooltip(hpc + "\n\nClick for the cluster connection");
                const float dot = theme::snap(px(6));
                const float cy = theme::snap((top + max.y) * 0.5f);
                dl->AddCircleFilled(ImVec2(x0 - px(8), cy), dot * 0.5f, hpcColor);
                widgets::drawTextIn(dl, ImVec2(x0, top), ImVec2(x0 + wd, max.y), shown, 11, hpcColor, Weight::Regular, 0.0f, 0.5f);
                limit = x0 - px(14) - px(24);
            }

            float x = min.x + px(14);
            auto item = [&](const std::string& s, ImU32 color = theme::kNeutral600) {
                if (s.empty()) return;
                const float wd = theme::textSize(s, 11).x;
                if (x + wd > limit) return;
                widgets::drawTextIn(dl, ImVec2(x, top), ImVec2(x + wd, max.y), s, 11, color, Weight::Regular, 0.0f, 0.5f);
                x += wd + px(24);
            };
            if (w.hasDataset()) {
                const DatasetMeta& m = w.dataset();
                item(m.shapeString());
                item(std::string(toString(m.sourceType)) + " \xE2\x86\x92 float32");
            } else {
                item("no dataset");
            }
            item("zoom " + viewer->zoomText());
            item(viewer->cursorText());

            double fraction = 0.0;
            std::string message;
            bool active = false;
            updateProgress(fraction, message, active);
            if (active) {
                // 160 px bar, 4 px tall, accent fill; then "53 % · ~40 s left · cellpose"
                const float bw = px(160), bh = theme::snap(px(4));
                if (x + bw < limit) {
                    const float cy = theme::snap((top + max.y) * 0.5f - bh * 0.5f);
                    dl->AddRectFilled(ImVec2(x, cy), ImVec2(x + bw, cy + bh), theme::kNeutral300);
                    dl->AddRectFilled(ImVec2(x, cy), ImVec2(x + bw * static_cast<float>(std::clamp(fraction, 0.0, 1.0)), cy + bh), theme::kAccent);
                    x += bw + px(8);
                }
                std::string text = format("%d %%", static_cast<int>(fraction * 100.0 + 0.5));
                const double elapsed = secondsSince(progressStart);
                if (fraction >= 0.03 && fraction < 1.0 && elapsed >= 2.0)
                    text += " \xC2\xB7 ~" + durationText(elapsed * (1.0 - fraction) / fraction) + " left";
                else if (elapsed >= 2.0) text += " \xC2\xB7 " + durationText(elapsed);
                if (!message.empty()) text += " \xC2\xB7 " + message;
                item(widgets::elideText(text, std::max(px(40), limit - x), 11), theme::kText);
            }

            // The last log line, for four seconds; the whole history is one
            // click (or the Window > Log shortcut) away.
            if (!logLine.empty() && secondsSince(logLineAt) < 4.0 && x < limit - px(60)) {
                const std::string shown = widgets::elideText(simplified(logLine), std::min(px(640), limit - x), 11);
                const float wd = theme::textSize(shown, 11).x;
                ImGui::SetCursorScreenPos(ImVec2(x, top));
                if (ImGui::InvisibleButton("##logline", ImVec2(std::max(wd, 1.0f), max.y - top))) self.showLog();
                const bool hovered = ImGui::IsItemHovered();
                if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                widgets::tooltip(simplified(logLine) + "\n\nClick to open the log (" + shortcutText(keys::logDock) + ")");
                widgets::drawTextIn(dl, ImVec2(x, top), ImVec2(x + wd, max.y), shown, 11, hovered ? theme::kAccentText : theme::kNeutral600,
                                    Weight::Regular, 0.0f, 0.5f);
                self.requestRedraw(2);   // so the line leaves when its time is up
            }
        }
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(3);
    }

    // --- dock windows -------------------------------------------------------------

    void App::Impl::drawDockWindows() {
        ImGuiViewport* vp = ImGui::GetMainViewport();
        dockspace = ImGui::GetID("SiriusDockSpace");
        if (!layoutBuilt || rebuildLayout) {
            // loaded in this frame's NewFrame: written back without the hidden tab bars
            if (!layoutBuilt && showHiddenTabBars()) ImGui::MarkIniSettingsDirty();
            // a saved arrangement (imgui.ini) is kept; without one, the design's
            if (rebuildLayout || ImGui::DockBuilderGetNode(dockspace) == nullptr) buildDefaultLayout(dockspace, vp->WorkSize);
            layoutBuilt = true;
            rebuildLayout = false;
        }
        // Where the panels asked to dock go, before the dockspace is laid out,
        // so a new split or size shows this frame. Only for a panel that is
        // shown: a target found for a hidden one would split the central node
        // again on every frame until it is shown.
        forgetLostDocks();
        // A panel asked to the front while another one is maximised would get
        // the keyboard out of sight, under the stand-in: the maximised one is
        // restored first.
        const std::array<bool, kPanelCount> focusing{false, focusOps || openAddMenu, false, focusAssistant, focusLog, focusDiag};
        for (std::size_t i = 0; i < kPanelCount; ++i)
            if (focusing[i] && !places[i].maximized) restoreMaximized(static_cast<Panel>(i));
        std::array<ImGuiID, kPanelCount> dockTo{};
        for (std::size_t i = 0; i < kPanelCount; ++i) {
            const bool* shown = shownFlag(static_cast<Panel>(i));
            if (places[i].request != Move::Dock || (shown && !*shown)) continue;
            dockTo[i] = dockTarget(static_cast<Panel>(i));
            if (!dockTo[i] && places[i].request == Move::Dock) {
                // no central node to split: the last resort is the default arrangement
                places[i].request = Move::None;
                rebuildLayout = true;
                wb().logLine(std::string("Layout reset to the default arrangement: ") + places[i].window + " had no place to dock back to.");
            }
        }
        followDiagnosticsCollapse();
        // Every dock shows its tab bar, even for one panel: a panel's tab is
        // how it is moved (dragged onto another dock, or out to float), so no
        // dock may hide it, and there is no window menu to hide it from.
        ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kNeutral900);
        ImGui::DockSpaceOverViewport(dockspace, vp, ImGuiDockNodeFlags_NoWindowMenuButton);
        ImGui::PopStyleColor();

        // Panels lay out their own margins: the docks have none.
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, px(120, 60));
        // A maximised panel covers the main window between the title bar and
        // the status bar.
        const ImVec2 workMin = vp->WorkPos;
        const ImVec2 workMax(vp->WorkPos.x + vp->WorkSize.x, vp->WorkPos.y + vp->WorkSize.y);

        // The viewer: a panel like the others, which the default arrangement
        // puts in the central node. It has no close box, as nothing would
        // show it again.
        bool covered = false;   // by the maximised diagnostics, this frame
        viewerInFront = false;
        viewerDrawn = false;
        ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBg);
        if (placeOf(Panel::Viewer).maximized) {
            if (beginStandIn(Panel::Viewer, nullptr, workMin, workMax)) {
                viewer->draw();
                viewerDrawn = true;
            }
            endStandIn(Panel::Viewer, workMin, workMax);
        } else {
            if (beginPanel(Panel::Viewer, nullptr, dockTo[static_cast<std::size_t>(Panel::Viewer)])) {
                // A tab about to come to the front is drawn a frame early, from
                // behind: Begin alone does not say the viewer is on screen.
                const ImGuiWindow* vw = ImGui::GetCurrentWindow();
                viewerInFront = vw->DockIsActive && vw->DockTabIsVisible && inDockspace(vw->DockNode, dockspace);
                // below its tab bar: the tabs of its dock stay usable under the cover
                viewerMin = ImGui::GetCursorScreenPos();
                viewerMax = ImVec2(ImGui::GetWindowPos().x + ImGui::GetWindowSize().x, ImGui::GetWindowPos().y + ImGui::GetWindowSize().y);
                covered = diagMaximized && showDiag && viewerInFront;
                if (!covered) viewer->draw();
                viewerDrawn = !covered;
            }
            ImGui::End();
            drawWindowControls(Panel::Viewer, ImGui::FindWindowByName(kViewerWindow), false);
        }
        ImGui::PopStyleColor();

        // A panel in its own window, or maximised in its stand-in; `focus` is
        // a request to give it the keyboard, which `body` may read too.
        const auto drawPanel = [&](Panel p, bool& focus, const auto& body) {
            Place& pl = placeOf(p);
            bool* open = shownFlag(p);
            if (!*open) {
                // hidden: its maximise, and a move asked for it as it was, lapse
                pl.maximized = false;
                pl.request = Move::None;
                focus = false;
                return;
            }
            if (focus) ImGui::SetNextWindowFocus();
            if (pl.maximized) {
                if (beginStandIn(p, open, workMin, workMax)) body();
                endStandIn(p, workMin, workMax);
            } else {
                if (beginPanel(p, open, dockTo[static_cast<std::size_t>(p)])) body();
                ImGui::End();
                drawWindowControls(p, ImGui::FindWindowByName(pl.window), false);
            }
            focus = false;
        };
        drawPanel(Panel::Ops, focusOps, [&] {
            if (openAddMenu) {
                openAddMenu = false;
                ops->openAddMenu();
            }
            ops->draw();
        });
        bool noFocus = false;
        drawPanel(Panel::Params, noFocus, [&] { params->draw(); });
        // Not while Parameters are maximised: their window, which it is placed
        // beside, is not drawn then and has no dock; it waits for Restore.
        if (showAssistant && !assistantPlaced && !placeOf(Panel::Assistant).maximized && !placeOf(Panel::Params).maximized) {
            // right of Parameters, the design's 330 px, the first time it is
            // shown and after the layout was rebuilt; not into Parameters
            // floating on their own (a dock of their own too), which it would
            // widen and split
            assistantPlaced = true;
            const bool place = forcePlaceAssistant || !ImGui::FindWindowSettingsByID(ImHashStr(kAssistantWindow));
            forcePlaceAssistant = false;
            if (ImGuiWindow* pw = ImGui::FindWindowByName(kParamsWindow)) {
                if (pw->DockNode && inDockspace(pw->DockNode, dockspace) && place) {
                    ImGuiID target = pw->DockNode->ID, side = 0, rest = 0;
                    forgetLostDocks();   // the dockspace was laid out since: the split may take a lost dock's id
                    const float total = pw->DockNode->Size.x + px(theme::kAssistantW);
                    ImGui::DockBuilderSetNodeSize(target, ImVec2(total, pw->DockNode->Size.y));
                    ImGui::DockBuilderSplitNode(target, ImGuiDir_Right, px(theme::kAssistantW) / std::max(total, 1.0f), &side, &rest);
                    ImGui::DockBuilderDockWindow(kAssistantWindow, side);
                    ImGui::DockBuilderFinish(dockspace);
                }
            }
        }
        drawPanel(Panel::Assistant, focusAssistant, [&] {
            if (focusAssistant) assistant->focusInput();
            assistant->draw();
        });
        // The log before the diagnostics: a dock tab that appears later
        // selects itself, and the diagnostics are the tab to start on.
        drawPanel(Panel::Log, focusLog, [&] {
            if (focusLog) log->showLatest();
            log->draw();
        });

        bool diagInFront = false;
        Place& diag = placeOf(Panel::Diag);
        // Floating under the cover, the panel's window is not drawn: the cover
        // shows the panel, and the window keeps its place for Restore.
        const bool diagUnderCover = covered && showDiag && diag.floating && diag.request == Move::None;
        if (diagUnderCover) {
            focusDiag = false;
        } else if (showDiag) {
            if (focusDiag) ImGui::SetNextWindowFocus();
            focusDiag = false;
            const bool toFront = diagFrontFrames > 0;
            // collapsed to its header, a floating diagnostics window is lower than a panel may otherwise be
            ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(px(120), diagCollapsed ? px(theme::kDiagnosticsHeaderH) : px(60)));
            const bool open = beginPanel(Panel::Diag, &showDiag, dockTo[static_cast<std::size_t>(Panel::Diag)]);
            ImGui::PopStyleVar();
            ImGuiWindow* dw = ImGui::GetCurrentWindow();
            // Behind another tab, the rectangle it had is the one last shown.
            diagInFront = open && (!dw->DockIsActive || dw->DockTabIsVisible);
            if (toFront) {
                // selected in its tab bar without taking the keyboard from the viewer
                if (dw->DockNode && dw->DockNode->TabBar) {
                    dw->DockNode->TabBar->NextSelectedTabId = dw->TabId;
                    dw->DockNode->SelectedTabId = dw->TabId;
                    --diagFrontFrames;   // counted once the tab bar exists
                }
                self.requestRedraw(2);
            }
            if (open) {
                diagInset = ImGui::GetCursorScreenPos().y - dw->Pos.y;
                if (!covered) diagnostics->draw();
            }
            ImGui::End();
            drawWindowControls(Panel::Diag, dw, false);
        } else {
            // hidden: a float or dock request made for the panel as it was lapses
            diagMaximized = false;
            diag.request = Move::None;
        }

        // Maximised: the diagnostics take the viewer's room below its tab bar
        // until they are restored, in their stand-in with its own tab bar. The
        // arrangement underneath is left as it is, and the tabs of the viewer's
        // dock stay usable. While the viewer is behind another tab, floating
        // or maximised itself, the main window has no room of the viewer's to
        // give: the diagnostics stay in their own window, still maximised, and
        // cover the viewer once it is docked there with its tab in front again.
        if (covered && showDiag) {
            // The cover takes the diagnostics' own room as well only when they
            // are docked right under the viewer; otherwise it keeps to the
            // viewer, and never spreads over Parameters or another monitor.
            const bool under = diagInFront && !diag.floating && diag.max.x > diag.min.x && std::abs(diag.min.y - viewerMax.y) <= px(4) &&
                               diag.min.x < viewerMax.x && diag.max.x > viewerMin.x;
            const ImVec2 min(viewerMin.x, viewerMin.y);
            const ImVec2 max(viewerMax.x, std::max(viewerMax.y, under ? diag.max.y : viewerMax.y));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBg);
            if (beginStandIn(Panel::Diag, &showDiag, min, max)) diagnostics->draw();
            endStandIn(Panel::Diag, min, max);
            ImGui::PopStyleColor();
        }
        ImGui::PopStyleVar(2);
        // Docks laid out this frame may have been merged away: forgotten
        // before the next frame can hand their ids to new docks.
        forgetLostDocks();

        help->draw();
    }

    // A panel's own window. It applies what the panel's Dock or Float control
    // asked (`dockTo`: the dock found for a Dock), and notes where the window
    // is now.
    bool App::Impl::beginPanel(Panel p, bool* open, ImGuiID dockTo) {
        Place& pl = placeOf(p);
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const ImGuiWindow* w = ImGui::FindWindowByName(pl.window);
        // Floating, a panel has the design's floating frame, 2 px of ink, as
        // the help window and the dialogs do (drawWindowControls draws it
        // round its dock, the floating window); docked, the rules between the
        // docks are its edges. A docked window given a border draws none but
        // keeps its content inside it on three sides: clear of the frame. The
        // border is fixed from Begin on, so where the window goes this frame
        // decides, or else where it was last frame.
        bool framed = w && !inDockspace(w->DockId ? ImGui::DockBuilderGetNode(w->DockId) : nullptr, dockspace);
        if (pl.request == Move::Float) {
            // Out of its dock into a window of its own near where it was: the
            // size it had, within reason (at least 480 x 360, at most 80 % of
            // the main window's work area), a little down and right of where
            // it was, inside the main window.
            const ImVec2 most(vp->WorkSize.x * 0.8f, vp->WorkSize.y * 0.8f);
            const ImVec2 least(std::min(px(480), most.x), std::min(px(360), most.y));
            const ImVec2 size(std::clamp(pl.max.x - pl.min.x, least.x, most.x), std::clamp(pl.max.y - pl.min.y, least.y, most.y));
            const ImVec2 room(std::max(vp->WorkPos.x, vp->WorkPos.x + vp->WorkSize.x - size.x),
                              std::max(vp->WorkPos.y, vp->WorkPos.y + vp->WorkSize.y - size.y));
            ImGui::SetNextWindowDockID(0, ImGuiCond_Always);
            ImGui::SetNextWindowSize(size, ImGuiCond_Always);
            ImGui::SetNextWindowPos(ImVec2(std::clamp(pl.min.x + px(40), vp->WorkPos.x, room.x), std::clamp(pl.min.y + px(40), vp->WorkPos.y, room.y)),
                                    ImGuiCond_Always);
            ImGui::SetNextWindowFocus();
            framed = true;
        } else if (pl.request == Move::Dock && dockTo) {
            ImGui::SetNextWindowDockID(dockTo, ImGuiCond_Always);
            framed = false;
        }
        pl.request = Move::None;
        ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, framed ? theme::crispPen(2) : 0.0f);
        ImGui::SetNextWindowClass(&panelClass());
        const bool shown = ImGui::Begin(pl.window, open, kPanelFlags);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        // Where it is docked is remembered on every frame, to dock it back there.
        const ImGuiWindow* cw = ImGui::GetCurrentWindow();
        pl.floating = !(cw->DockIsActive && cw->DockNode && inDockspace(cw->DockNode, dockspace));
        if (!pl.floating) pl.dock = cw->DockId;
        // Floating, the window the user resizes is its dock's, which Dear
        // ImGui begins before this with its own minimum size (32 px), not the
        // panels' one pushed around this: the dock is held to the panels' from
        // the next frame on.
        if (pl.floating && cw->DockIsActive && cw->DockNode) {
            const ImGuiDockNode* root = ImGui::DockNodeGetRootNode(cw->DockNode);
            const ImVec2 least = ImGui::GetStyle().WindowMinSize;
            if (root->HostWindow && (root->Size.x < least.x - 0.5f || root->Size.y < least.y - 0.5f))
                ImGui::DockBuilderSetNodeSize(root->ID, ImMax(root->Size, least));
        }
        // Behind another tab, the rectangle it had is the one last shown.
        if (shown) {
            pl.min = cw->Pos;
            pl.max = ImVec2(cw->Pos.x + cw->Size.x, cw->Pos.y + cw->Size.y);
        }
        return shown;
    }

    // A maximised panel is drawn in a stand-in window pinned over `min` -
    // `max` (the main window's work area; the viewer's room for the
    // diagnostics), on top of the rest, with a tab bar of its own for its
    // controls. Its own window is not drawn meanwhile, and keeps its place.
    bool App::Impl::beginStandIn(Panel p, bool* open, ImVec2 min, ImVec2 max) {
        // Until Dear ImGui gives its dock a window, the stand-in is a window
        // on its own for a frame, placed here; after that endStandIn places
        // its dock.
        ImGui::SetNextWindowPos(min, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(std::max(max.x - min.x, 1.0f), std::max(max.y - min.y, 1.0f)), ImGuiCond_Always);
        ImGui::SetNextWindowClass(&standInClass());
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        const bool shown = ImGui::Begin((std::string(placeOf(p).window) + kStandInSuffix).c_str(), open,
                                        kPanelFlags | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove);
        ImGui::PopStyleVar();
        return shown;
    }

    void App::Impl::endStandIn(Panel p, ImVec2 min, ImVec2 max) {
        ImGuiWindow* w = ImGui::GetCurrentWindow();
        ImGui::End();
        // Its dock is the floating window: pinned from the next frame on, as
        // the main window is resized; dragging its tab moves it nowhere.
        if (w->DockNode && w->DockNode->IsRootNode() && max.x > min.x && max.y > min.y) {
            ImGui::DockBuilderSetNodePos(w->DockNode->ID, min);
            ImGui::DockBuilderSetNodeSize(w->DockNode->ID, ImVec2(max.x - min.x, max.y - min.y));
        }
        drawWindowControls(p, w, true);
    }

    // The window controls at the right end of a dock's tab bar: Dock, Float
    // and Maximise, for the panel whose tab is in front there (in a dock the
    // log shares with the diagnostics, the set follows the tab). The control
    // of where the panel is now -- docked in the main window, floating, or
    // maximised -- is drawn active, as the diagnostics header drew its mode.
    // A floating dock also gets the design's 2 px ink frame here, round its
    // tab bar and its edges: it is drawn by the dock's own window, on top of
    // its tab bar, and the panels in it keep their content inside the frame
    // (beginPanel).
    void App::Impl::drawWindowControls(Panel p, ImGuiWindow* panelWindow, bool standIn) {
        if (!panelWindow || !panelWindow->Active || !panelWindow->DockNode || panelWindow->DockNode->VisibleWindow != panelWindow) return;
        ImGuiDockNode* node = panelWindow->DockNode;
        if (!ImGui::DockNodeBeginAmendTabBar(node)) return;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImGuiDockNode* root = ImGui::DockNodeGetRootNode(node);
        if (!standIn && root->HostWindow && !inDockspace(node, dockspace)) {
            const ImGuiWindow* host = root->HostWindow;
            widgets::crispRect(dl, host->Pos, ImVec2(host->Pos.x + host->Size.x, host->Pos.y + host->Size.y), theme::kText, 2);
        }

        const Place& pl = placeOf(p);
        const bool maximized = p == Panel::Diag ? diagMaximized : pl.maximized;
        const bool floating = !maximized && pl.floating;
        const bool docked = !maximized && !floating;
        // Drawn over the tab's own rectangle, which is left empty: an icon and
        // its word, so the three read apart at a glance. Idle they are quiet
        // text on the tab bar, hovered they get the surface colour, and the
        // one for the panel's current state is filled with the accent.
        ImGui::PushStyleColor(ImGuiCol_TabHovered, theme::kTransparent);
        // Dear ImGui lays the trailing tabs out right after the other tabs
        // (further left only when those overflow). Each control is moved to
        // the right end before it is submitted, as far from it as the first
        // tab is from the left one: from this frame's layout and tab bar, so
        // the controls keep to the edge while a dock is resized, and never
        // left of where the layout put them, onto the tabs.
        ImGuiTabBar* bar = node->TabBar;
        const float gap = ImGui::GetStyle().ItemInnerSpacing.x;
        const float margin = std::max(bar->BarRect.Min.x - node->Pos.x, px(4));
        const float end = bar->BarRect.GetWidth() - margin;   // tab offsets are from the bar's left end
        constexpr float kLabelPx = 12;
        const float iconSide = theme::snap(px(14)), padX = px(8), iconGap = px(5);
        struct Control {
            const char* id;
            Icon icon;
            const char* label;
            bool active;
            const char* tip;
        };
        const char* maximize = p == Panel::Diag ? "Maximise over the viewer" : "Maximise over the main window";
        const Control controls[3] = {
            {"##dock", Icon::Dock, "Dock", docked, docked ? "Docked in the main window" : "Dock in the main window"},
            {"##float", Icon::Float, "Float", floating, floating ? "Floating outside the main window" : "Float in a window of its own"},
            {"##maximize", Icon::Maximize, maximized ? "Restore" : "Maximize", maximized, maximized ? "Restore" : maximize},
        };
        // Labelled while the dock's own tabs keep their full width beside
        // them; in a narrower dock only the current state keeps its word, and
        // in the narrowest the three are icons (their tooltips still name them).
        float tabsW = 0.0f;
        for (const ImGuiTabItem& tab : bar->Tabs)
            if (!(tab.Flags & ImGuiTabItemFlags_SectionMask_)) tabsW += tab.ContentWidth + ImGui::GetStyle().FramePadding.x * 2 + gap;
        const auto widthsFor = [&](int mode, float* out) {   // 0: all labelled, 1: the active one, 2: none
            float total = 0.0f;
            for (int i = 0; i < 3; ++i) {
                const bool label = mode == 0 || (mode == 1 && controls[i].active);
                out[i] = theme::snap(label ? padX * 2 + iconSide + iconGap + theme::textSize(controls[i].label, kLabelPx, Weight::SemiBold).x
                                           : px(26));
                total += out[i] + gap;
            }
            return total;
        };
        float widths[3];
        int mode = 0;
        while (mode < 2 && tabsW + widthsFor(mode, widths) + margin * 2 > bar->BarRect.GetWidth()) ++mode;
        widthsFor(mode, widths);
        bool pressed[3] = {false, false, false};
        for (int i = 0; i < 3; ++i) {
            const Control& c = controls[i];
            float right = 0.0f;   // the width of the controls right of this one
            for (int j = i + 1; j < 3; ++j) right += widths[j] + gap;
            if (ImGuiTabItem* tab = ImGui::TabBarFindTabByID(bar, ImGui::GetID(c.id)))
                tab->Offset = std::max(tab->Offset, std::floor(end - widths[i] - right));
            ImGui::SetNextItemWidth(widths[i]);
            pressed[i] = ImGui::TabItemButton(c.id, ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip);
            const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
            if (b.x - a.x < 1.0f) {   // its first frame: taken into the tab bar, not laid out yet
                pressed[i] = false;
                continue;
            }
            const bool hovered = ImGui::IsItemHovered();
            if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            const float h = theme::snap(px(22));
            const ImVec2 min(a.x, theme::snap((a.y + b.y - h) * 0.5f));
            const ImVec2 max(b.x, min.y + h);
            if (c.active) dl->AddRectFilled(min, max, theme::kAccent);
            else if (hovered) dl->AddRectFilled(min, max, theme::kSurface);
            const ImU32 ink = c.active ? theme::kBg : hovered ? theme::kText
                                                              : theme::kNeutral600;
            const float cy = (min.y + max.y) * 0.5f;
            if (mode == 0 || (mode == 1 && c.active)) {
                drawIcon(dl, ImVec2(min.x + padX + iconSide * 0.5f, cy), iconSide, c.icon, ink, px(1.5f));
                widgets::drawTextIn(dl, ImVec2(min.x + padX + iconSide + iconGap, min.y), ImVec2(max.x, max.y), c.label, kLabelPx, ink,
                                    Weight::SemiBold, 0.0f, 0.5f);
            } else {
                drawIcon(dl, ImVec2((min.x + max.x) * 0.5f, cy), iconSide, c.icon, ink, px(1.5f));
            }
            widgets::tooltip(c.tip);
        }
        if (pressed[0]) dockPanel(p);
        if (pressed[1]) floatPanel(p);
        if (pressed[2]) toggleMaximized(p);
        ImGui::PopStyleColor();
        ImGui::DockNodeEndAmendTabBar();
    }

    // The window controls' commands. The diagnostics' are App's (the Window
    // menu and the tool calls use them too), their maximise the cover over
    // the viewer; a change of mode shows them expanded, as a collapsed panel
    // would be a strip in its new place.
    void App::Impl::dockPanel(Panel p) {
        Place& pl = placeOf(p);
        if (p == Panel::Diag) {
            if (diagMaximized || pl.floating) diagnostics->setCollapsed(false);
            self.dockDiagnostics();
            return;
        }
        // maximised, it goes back to its dock if it had one; floating, it docks
        pl.maximized = false;
        if (pl.floating) pl.request = Move::Dock;
        self.requestRedraw();
    }

    void App::Impl::floatPanel(Panel p) {
        Place& pl = placeOf(p);
        if (p == Panel::Diag) {
            if (diagMaximized || !pl.floating) {
                diagnostics->setCollapsed(false);
                self.floatDiagnostics();
            }
            return;
        }
        pl.maximized = false;
        if (!pl.floating) pl.request = Move::Float;
        self.requestRedraw();
    }

    void App::Impl::toggleMaximized(Panel p) {
        Place& pl = placeOf(p);
        if (p == Panel::Diag) {
            // Floating, the panel keeps its window, not drawn under the cover,
            // and Restore shows it there again.
            diagnostics->setCollapsed(false);
            self.setDiagnosticsMaximized(!diagMaximized);
            return;
        }
        pl.maximized = !pl.maximized;
        pl.request = Move::None;
        if (pl.maximized) {
            restoreMaximized(p);
            diagMaximized = false;
        }
        self.requestRedraw();
    }

    // --- dialogs ----------------------------------------------------------------------

    void App::Impl::drawDialogBody(Dialog& d) {
        // title row: 20 px / 800 and the close box
        const float w = ImGui::GetContentRegionAvail().x;
        const ImVec2 start = ImGui::GetCursorScreenPos();
        widgets::heading(d.title(), theme::kH4Px);
        const ImVec2 after = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos(ImVec2(start.x + w - px(24), start.y));
        widgets::GlyphOpts g;
        g.borderless = true;
        g.tooltip = "Close (Esc)";
        if (widgets::glyphButton("##closeDialog", Icon::Close, 24, g)) {
            if (d.canClose(self)) d.close();
        }
        ImGui::SetCursorScreenPos(after);
        widgets::vspace(4);
        d.draw(self);
    }

    void App::Impl::drawDialogs(std::size_t index) {
        if (index >= dialogs.size()) return;
        const std::shared_ptr<Dialog> d = dialogs[index];   // keeps it alive through its own close
        const std::string id = d->title() + "###dialog" + std::to_string(reinterpret_cast<std::uintptr_t>(d.get()));
        const ImVec2 size = d->size();
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoDocking;
        if (!d->resizable()) flags |= ImGuiWindowFlags_NoResize;
        if (size.y <= 0.0f) flags |= ImGuiWindowFlags_AlwaysAutoResize;
        // The room a dialog may take: the main window's work area while it is
        // in there (it opens there), the work area of its monitor once it has
        // a window of its own, so another monitor neither caps it nor shrinks
        // it with the main window.
        ImVec2 room(vp->WorkSize.x, vp->WorkSize.y), roomPos(vp->WorkPos.x, vp->WorkPos.y);
        if (d->appeared_ && (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)) {
            const ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
            if (ImGuiWindow* w = ImGui::FindWindowByName(id.c_str());
                w && w->Viewport && w->Viewport != vp && w->Viewport->PlatformMonitor >= 0 && w->Viewport->PlatformMonitor < pio.Monitors.Size) {
                room = pio.Monitors[w->Viewport->PlatformMonitor].WorkSize;
                roomPos = pio.Monitors[w->Viewport->PlatformMonitor].WorkPos;
            }
        }
        const float maxH = std::max(room.y - px(40), px(120));
        if (!d->appeared_) {
            ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.5f), ImGuiCond_Appearing,
                                    ImVec2(0.5f, 0.5f));
            if (size.y > 0.0f) ImGui::SetNextWindowSize(ImVec2(px(size.x), std::min(px(size.y), maxH)), ImGuiCond_Appearing);
        } else if (ImGuiWindow* w = ImGui::FindWindowByName(id.c_str())) {
            // A dialog placed when it appeared grows downwards as its content
            // fills in (an auto-sized one, or one whose rows come later): kept
            // inside its room, or its buttons end up below the screen with the
            // dialog holding the input of the window behind it.
            const ImVec2 lo = roomPos, hi(roomPos.x + room.x, roomPos.y + room.y);
            ImVec2 pos = w->Pos;
            pos.x = std::max(lo.x, std::min(pos.x, hi.x - w->Size.x));
            pos.y = std::max(lo.y, std::min(pos.y, hi.y - w->Size.y));
            if (std::abs(pos.x - w->Pos.x) > 0.5f || std::abs(pos.y - w->Pos.y) > 0.5f) ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
        }
        if (size.y <= 0.0f) {
            ImGui::SetNextWindowSizeConstraints(ImVec2(px(size.x), 0.0f), ImVec2(px(size.x), maxH));
        } else if (d->resizable()) {
            // never a minimum above the maximum: the smaller room wins
            const ImVec2 lo(std::min(px(360), room.x), std::min(px(240), room.y));
            ImGui::SetNextWindowSizeConstraints(lo, ImVec2(std::max(lo.x, room.x), std::max(lo.y, room.y)));
        }
        ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBg);
        ImGui::PushStyleColor(ImGuiCol_PopupBg, theme::kBg);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, theme::crispPen(2));
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(18, 16));
        bool begun = false;
        if (d->modal()) {
            if (!d->appeared_) ImGui::OpenPopup(id.c_str());
            begun = ImGui::BeginPopupModal(id.c_str(), nullptr, flags);
            // Dear ImGui closed it (it never does for a modal, but a popup of
            // a closed parent goes with its parent)
            if (!begun && d->appeared_ && !ImGui::IsPopupOpen(id.c_str())) d->open_ = false;
        } else {
            if (d->raise_) ImGui::SetNextWindowFocus();
            d->raise_ = false;
            ImGui::Begin(id.c_str(), nullptr, flags);
            begun = true;
            if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) dialogHasFocus = true;
        }
        d->appeared_ = true;
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(3);
        if (begun) {
            // On top: no modal is open over this dialog. Every modal later in
            // the stack is drawn nested in an earlier modal's popup, whatever
            // non-modal windows sit between them; a non-modal dialog is under
            // any modal at all.
            const auto isModal = [](const std::shared_ptr<Dialog>& x) { return x->modal(); };
            const bool top = d->modal() ? std::none_of(dialogs.begin() + static_cast<std::ptrdiff_t>(index) + 1, dialogs.end(), isModal)
                                        : std::none_of(dialogs.begin(), dialogs.end(), isModal);
            d->onTop_ = top;
            if (d->modal() && top && ImGui::IsKeyPressed(ImGuiKey_Escape, false) && !ImGui::IsAnyItemActive()) {
                // Escape dismisses a dropdown or menu the dialog has open (no
                // keyboard navigation does it for us) before the dialog itself.
                if (ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) ImGui::ClosePopupToLevel(GImGui->OpenPopupStack.Size - 1, true);
                else if (d->canClose(self)) d->close();
            }
            ImGui::PushID(d.get());
            drawDialogBody(*d);
            ImGui::PopID();
            if (d->modal()) {
                // the dialogs above this one are its children, as Dear ImGui stacks popups
                std::size_t next = index + 1;
                while (next < dialogs.size() && !dialogs[next]->modal()) ++next;
                if (d->isOpen()) drawDialogs(next);
                if (!d->isOpen()) ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            } else {
                ImGui::End();
            }
        }
    }

    // --- shortcuts ------------------------------------------------------------------------

    void App::Impl::handleShortcuts() {
        ImGuiIO& io = ImGui::GetIO();
        bool modal = dialogHadFocus;
        for (const auto& d : dialogs) modal = modal || d->modal();
        // A key the focused widget uses itself stays with the widget: text
        // being typed, a field being edited, a popup that is open.
        const bool typing = io.WantTextInput || ImGui::IsAnyItemActive();
        const bool popup = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        for (Action& a : actions) {
            if (!a.run) continue;
            for (ImGuiKeyChord chord : a.keys) {
                const bool plain = (chord & (ImGuiMod_Ctrl | ImGuiMod_Alt | ImGuiMod_Super)) == 0;
                if (modal) continue;
                if (plain && (typing || popup)) continue;
                // Ctrl+C / V / Z / Y belong to the text field while one has the keyboard
                if (!plain && io.WantTextInput) continue;
                if (popup && chord != ImGuiKey_Escape) continue;
                if (std::find(claimedKeys.begin(), claimedKeys.end(), chord) != claimedKeys.end()) continue;
                // A plain key a widget has taken (Dear ImGui's key ownership: a
                // focused slider owns the arrows it steps with) stays with it.
                if (plain ? !ImGui::IsKeyChordPressed(chord, ImGuiInputFlags_None, ImGuiKeyOwner_NoOwner) : !ImGui::IsKeyChordPressed(chord)) continue;
                if (a.enabled && !a.enabled()) continue;
                self.defer(a.run);
                return;   // one key, one action
            }
        }
    }

    void App::Impl::captureScreenshot() {
        if (pendingScreenshot.empty() || !window) return;
        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        screenshotOk = false;
        if (w > 0 && h > 0) {
            std::vector<std::uint8_t> flipped(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4), image(flipped.size());
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadBuffer(GL_BACK);
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, flipped.data());
            const std::size_t row = static_cast<std::size_t>(w) * 4;
            for (int y = 0; y < h; ++y)
                std::copy_n(flipped.data() + static_cast<std::size_t>(h - 1 - y) * row, row, image.data() + static_cast<std::size_t>(y) * row);
            for (std::size_t i = 3; i < image.size(); i += 4) image[i] = 255;
            screenshotOk = writePng(pendingScreenshot, image.data(), w, h);
        }
        pendingScreenshot.clear();
    }

    void App::Impl::finishClose() {
        const bool run = bridge().running(), task = bridge().taskRunning();
        if (run) bridge().cancelRun();
        if (task) bridge().cancelTask();
        closing = true;
    }

    // -------------------------------------------------------------------------
    // App
    // -------------------------------------------------------------------------

    App::App(Bridge& bridge, ToolApi& tools, WorkerLauncher& launcher)
        : impl_(std::make_unique<Impl>(*this)), bridge_(bridge), tools_(tools), launcher_(launcher) {}

    App::~App() {
        Impl& d = *impl_;
        if (d.imguiReady) {
            // A scripted run (unattended) showed every panel and never the
            // user's layout: it leaves their panels and geometry alone.
            if (d.window && !d.unattended) {
                settings().set("window/showOperations", d.showOps);
                settings().set("window/showParameters", d.showParams);
                settings().set("window/showDiagnostics", d.showDiag);
                settings().set("window/showLog", d.showLog);
                int w = 0, h = 0;
                glfwGetWindowSize(d.window, &w, &h);
                const bool maximized = glfwGetWindowAttrib(d.window, GLFW_MAXIMIZED) != 0;
                settings().set("window/maximized", maximized);
                if (!maximized && w > 200 && h > 200) {
                    // Back to design pixels by the factor init() sizes the window
                    // by (not the user's zoom, or the window would shrink or grow
                    // every session): that of the monitor it is on now.
                    const float f = windowScale(d.window);
                    settings().set("window/width", static_cast<int>(std::lround(static_cast<float>(w) / f)));
                    settings().set("window/height", static_cast<int>(std::lround(static_cast<float>(h) / f)));
                }
            }
            settings().save();
            // The next session opens the panel expanded: its dock gets its
            // height back before Dear ImGui writes the layout (DestroyContext).
            if (d.diagCollapsed) d.fitDiagnosticsToCollapse(false);
        }
        bridge_.setWaker(nullptr);
        http::setGuiPoster(nullptr);
        // the panels own textures: they go while the context is current
        d.dialogs.clear();
        d.plugins.reset();
        d.help.reset();
        d.log.reset();
        d.assistant.reset();
        d.diagnostics.reset();
        d.params.reset();
        d.ops.reset();
        d.viewer.reset();
        if (d.imguiReady) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImPlot::DestroyContext();
            ImGui::DestroyContext();
        }
        if (d.window) glfwDestroyWindow(d.window);
        if (d.glfwReady) glfwTerminate();
    }

    GLFWwindow* App::window() const noexcept { return impl_->window; }
    bool App::unattended() const noexcept { return impl_->unattended; }
    void App::setUnattended(bool on) { impl_->unattended = on; }
    void App::setExitCode(int code) { impl_->exitCode = code; }
    int App::exitCode() const noexcept { return impl_->exitCode; }
    std::uint64_t App::frameCount() const noexcept { return impl_->frames; }

    void App::claimKey(ImGuiKeyChord chord) { impl_->claimingKeys.push_back(chord); }

    bool App::init(const InitOptions& options) {
        Impl& d = *impl_;
        d.unattended = options.unattended;
        d.visible = options.visible;
        glfwSetErrorCallback([](int code, const char* text) { std::fprintf(stderr, "glfw error %d: %s\n", code, text ? text : ""); });
        if (!glfwInit()) {
            std::fprintf(stderr, "sirius-app: cannot initialise the window system (no display?)\n");
            return false;
        }
        d.glfwReady = true;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);        // shown once it is sized and placed
        // A scripted run (a screenshot, a smoke test) must not take the keyboard
        // from whatever the user is doing while it runs.
        glfwWindowHint(GLFW_FOCUS_ON_SHOW, d.unattended ? GLFW_FALSE : GLFW_TRUE);
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_FALSE);
        // The window's app id on Wayland and its WM_CLASS on X11: how a desktop
        // matches the window to sirius-app.desktop for its icon and name.
        glfwWindowHintString(GLFW_WAYLAND_APP_ID, "sirius-app");
        glfwWindowHintString(GLFW_X11_CLASS_NAME, "sirius-app");
        glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "sirius-app");

        // The design's 1600 x 960 at this monitor's scale, inside its work area.
        // Where the framebuffer carries the density (Wayland, macOS) the window
        // is sized in logical units already, and monitorScale is 1.
        const bool wayland = glfwGetPlatform() == GLFW_PLATFORM_WAYLAND;
        float scale = 1.0f;
        int mx = 0, my = 0, mw = 0, mh = 0;
        if (GLFWmonitor* monitor = glfwGetPrimaryMonitor()) {
            scale = monitorScale(monitor);
            glfwGetMonitorWorkarea(monitor, &mx, &my, &mw, &mh);
            if (wayland) {
                // Wayland gives the work area in the output's pixels: logical units, for the window.
                float xs = 1.0f, ys = 1.0f;
                glfwGetMonitorContentScale(monitor, &xs, &ys);
                if (xs > 0.0f) {
                    mw = static_cast<int>(std::lround(static_cast<float>(mw) / xs));
                    mh = static_cast<int>(std::lround(static_cast<float>(mh) / xs));
                }
            }
        }
        int dw = options.width, dh = options.height;
        // --size wins over the size (and the maximized state) the last session left
        if (!d.unattended && !options.sizeGiven) {
            dw = std::max(640, settings().getInt("window/width", dw));
            dh = std::max(480, settings().getInt("window/height", dh));
        }
        int w = static_cast<int>(std::lround(static_cast<float>(dw) * scale));
        int h = static_cast<int>(std::lround(static_cast<float>(dh) * scale));
        if (mw > 0 && mh > 0 && !d.unattended) {
            w = std::min(w, mw);
            h = std::min(h, mh - 40);   // the title bar is outside the client area
        }
        d.window = glfwCreateWindow(w, h, "SIRIUS", nullptr, nullptr);
        if (!d.window) {
            std::fprintf(stderr, "sirius-app: cannot create a window with an OpenGL 3.3 context\n");
            return false;
        }
        {
            // The icon in every size there is a rendition of (app/resources/icons),
            // beside the executable or in an installed tree; Wayland ignores it
            // and takes the .desktop entry's.
            std::string dir = besideApplication("icons");
            if (dir.empty()) dir = installedDataDirectory("icons");
            std::vector<std::vector<std::uint8_t>> pixels;
            std::vector<GLFWimage> images;
            for (int px : {16, 24, 32, 48, 64, 128, 256}) {
                std::vector<std::uint8_t> rgba;
                int iw = 0, ih = 0;
                if (dir.empty() || !readImage(dir + format("/sirius-app-%d.png", px), rgba, iw, ih)) continue;
                pixels.push_back(std::move(rgba));
                images.push_back(GLFWimage{iw, ih, pixels.back().data()});
            }
            // after the loop: the vector of pixels no longer moves
            for (std::size_t i = 0; i < images.size(); ++i) images[i].pixels = pixels[i].data();
            if (!images.empty() && glfwGetPlatform() != GLFW_PLATFORM_WAYLAND)
                glfwSetWindowIcon(d.window, static_cast<int>(images.size()), images.data());
        }
        // Wayland has no window positions: the compositor places the window.
        if (mw > 0 && mh > 0 && !wayland) glfwSetWindowPos(d.window, mx + std::max(0, (mw - w) / 2), my + std::max(32, (mh - h) / 2));
        d.contentScale = windowScale(d.window);
        glfwMakeContextCurrent(d.window);
        glfwSwapInterval(1);
        if (!gladLoadGL(glfwGetProcAddress)) {
            std::fprintf(stderr, "sirius-app: cannot load OpenGL\n");
            return false;
        }
        glfwSetWindowUserPointer(d.window, this);
        // Ours first: the Dear ImGui backend chains to the callbacks it finds.
        glfwSetWindowCloseCallback(d.window, [](GLFWwindow* win) {
            glfwSetWindowShouldClose(win, GLFW_FALSE);   // the application decides
            if (auto* app = static_cast<App*>(glfwGetWindowUserPointer(win))) app->defer([app] { app->requestClose(); });
        });
        // Kept for the windows of floating panels as well (below).
        static GLFWdropfun onDrop = nullptr;
        onDrop = [](GLFWwindow* win, int count, const char** paths) {
            auto* app = static_cast<App*>(glfwGetWindowUserPointer(win));
            if (!app || count <= 0) return;
            std::vector<std::string> list;
            for (int i = 0; i < count; ++i) list.emplace_back(paths[i]);
            // Opening reads the file and can raise a dialog: both belong after
            // the drop has been answered.
            app->defer([app, list] { app->dropPaths(list); });
        };
        glfwSetDropCallback(d.window, onDrop);
        glfwSetWindowContentScaleCallback(d.window, [](GLFWwindow* win, float, float) {
            if (auto* app = static_cast<App*>(glfwGetWindowUserPointer(win))) app->defer([app] { app->impl_->rescale(); });
        });
        glfwSetWindowRefreshCallback(d.window, [](GLFWwindow* win) {
            // The window is being resized or uncovered while GLFW is in the
            // frame loop's poll (Windows runs its modal size / move loop in
            // there): draw, rather than leave a stale frame stretched over it.
            // Not while a native file dialog's loop runs a deferred action, and
            // never inside a frame.
            auto* app = static_cast<App*>(glfwGetWindowUserPointer(win));
            if (!app) return;
            Impl& s = *app->impl_;
            if (!s.imguiReady || !s.inPoll || s.closing || GImGui->WithinFrameScope || glfwGetWindowAttrib(win, GLFW_ICONIFIED)) return;
            // Nothing may unwind through GLFW and the window procedure: an
            // error is caught here and its frame finished (abandonFrame).
            try {
                s.drawFrame(true);
            } catch (const std::exception& e) {
                s.abandonFrame(e.what());
            } catch (...) {
                s.abandonFrame("unknown exception");
            }
        });

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        d.viewports = !d.unattended && settings().getBool("ui/floatingWindows", true);
        d.updateViewports();
        io.ConfigWindowsMoveFromTitleBarOnly = false;
        io.ConfigDockingWithShift = false;
        io.ConfigInputTextCursorBlink = true;
        io.ConfigDragClickToInputText = false;
        static std::string iniPath;   // Dear ImGui keeps the pointer
        iniPath = settings().layoutPath();
        platform::makePath(settings().directory());
        io.IniFilename = d.unattended ? nullptr : iniPath.c_str();
        io.LogFilename = nullptr;

        theme::loadFonts();
        d.applyScale();
        ImGui_ImplGlfw_InitForOpenGL(d.window, true);
        ImGui_ImplOpenGL3_Init("#version 330 core");
        {
            // GLFW accepts files dropped on every window, so a panel floating in
            // a window of its own (the viewer, say) takes them as the main
            // window does, rather than showing the drop cursor for nothing.
            static void (*createWindow)(ImGuiViewport*) = nullptr;
            ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
            createWindow = pio.Platform_CreateWindow;
            if (createWindow) {
                pio.Platform_CreateWindow = [](ImGuiViewport* v) {
                    createWindow(v);
                    auto* win = static_cast<GLFWwindow*>(v->PlatformHandle);
                    auto* mainWindow = static_cast<GLFWwindow*>(ImGui::GetMainViewport()->PlatformHandle);
                    if (!win || !mainWindow) return;
                    glfwSetWindowUserPointer(win, glfwGetWindowUserPointer(mainWindow));
                    glfwSetDropCallback(win, onDrop);
                };
            }
        }
        d.imguiReady = true;

        d.showOps = settings().getBool("window/showOperations", true);
        d.showParams = settings().getBool("window/showParameters", true);
        d.showDiag = settings().getBool("window/showDiagnostics", true);
        d.showLog = settings().getBool("window/showLog", true);
        if (d.unattended) d.showOps = d.showParams = d.showDiag = d.showLog = true;

        bridge_.setWaker([] { glfwPostEmptyEvent(); });
        http::setGuiPoster([this](std::function<void()> fn) { bridge_.post(std::move(fn)); });

        d.viewer = std::make_unique<Viewer>(*this);
        d.ops = std::make_unique<OpsPanel>(*this);
        d.params = std::make_unique<ParamsPanel>(*this);
        d.diagnostics = std::make_unique<DiagnosticsPanel>(*this);
        d.assistant = std::make_unique<AssistantPanel>(*this);
        d.log = std::make_unique<LogPanel>(*this);
        d.help = std::make_unique<HelpWindow>(*this);
        d.buildActions();

        // what has to happen once, when something happens
        bridge_.logged.connect([this](const std::string& line) {
            impl_->logLine = line;
            impl_->logLineAt = Clock::now();
            requestRedraw();
        });
        bridge_.runFinished.connect([this](bool ok, const std::string& error) {
            // A cancelled run arrives with an empty error, so there is no
            // message text to recognise here.
            if (!ok && !error.empty() && !impl_->unattended) message("Run failed", error);
            requestRedraw();
        });
        bridge_.taskFinished.connect([this](bool ok, const std::string& error) {
            if (!ok && !error.empty() && !impl_->unattended) message(bridge_.taskLabel(), error);
            requestRedraw();
        });
        bridge_.selectionChanged.connect([this] {
            if (impl_->help->visible()) showHelpForSelected();
        });
        for (Signal<>* s : {&bridge_.datasetChanged, &bridge_.pipelineChanged, &bridge_.viewStateChanged, &bridge_.outputsChanged,
                            &bridge_.historyChanged, &bridge_.backendChanged, &bridge_.runStateChanged, &bridge_.viewedStepChanged,
                            &bridge_.operationsChanged})
            s->connect([this] { requestRedraw(); });
        bridge_.stepChanged.connect([this](int) { requestRedraw(); });
        bridge_.labelsChanged.connect([this](StepId) { requestRedraw(); });

        d.savedRevision = wb().history().revision();
        if (d.visible) {
            if (!d.unattended && !options.sizeGiven && settings().getBool("window/maximized", false)) glfwMaximizeWindow(d.window);
            glfwShowWindow(d.window);
        }
        return true;
    }

    void App::requestRedraw(int frames) {
        impl_->redrawFrames = std::max(impl_->redrawFrames, frames);
        if (impl_->window) glfwPostEmptyEvent();
    }

    void App::defer(std::function<void()> action) {
        if (!action) return;
        impl_->deferred.push_back(std::move(action));
        requestRedraw();
    }

    void App::waitUntil(const std::function<bool()>& done) {
        while (!done() && !impl_->closing) {
            requestRedraw(1);
            if (!frame()) break;
        }
    }

    void App::settle() {
        for (int i = 0; i < 3 && !impl_->closing; ++i) {
            requestRedraw(1);
            frame();
        }
        // A dataset opens on a worker thread and is installed when that task
        // ends: the next scripted step waits for it, as a user would for the
        // progress bar, rather than find no dataset yet.
        const auto deadline = Clock::now() + std::chrono::seconds(600);
        waitUntil([&] { return !bridge_.taskRunning() || Clock::now() >= deadline; });
        for (int i = 0; i < 2 && !impl_->closing; ++i) {
            requestRedraw(1);
            frame();
        }
    }

    int App::run() {
        while (frame()) {
        }
        return impl_->exitCode;
    }

    bool App::frame() {
        Impl& d = *impl_;
        if (d.closing || !d.window) return false;
        ++d.frameDepth;

        // Sleep while nothing moves; wake for input, for a worker thread's
        // results (Bridge::wake) and for what animates.
        const bool animating = d.redrawFrames > 0 || bridge_.running() || bridge_.taskRunning() ||
                               (d.viewer && d.viewerDrawn && d.viewer->animating()) || (d.assistant && d.assistant->busy()) ||
                               !d.deferred.empty();
        // The platform layer is told too: a dialog opened by a refresh
        // callback's frame in here must not poll GLFW when it closes.
        d.inPoll = true;
        platform::setWithinEventPoll(true);
        if (animating || d.frameDepth > 1) glfwPollEvents();
        else glfwWaitEventsTimeout(0.25);
        d.inPoll = false;
        platform::setWithinEventPoll(false);
        if (d.redrawFrames > 0) --d.redrawFrames;

        bridge_.update();
        if (d.cluster) d.cluster->frame();
        if (!d.deferred.empty()) {
            std::vector<std::function<void()>> actions;
            actions.swap(d.deferred);
            for (auto& a : actions) {
                if (d.closing) break;
                try {
                    a();
                } catch (const std::exception& e) {
                    wb().logLine(std::string("Error: ") + e.what());
                }
            }
        }
        if (d.closing) {
            --d.frameDepth;
            return false;
        }
        // A native file dialog that could not be opened (no file chooser
        // portal, no GTK) returned as though cancelled: say why, once, whatever
        // opened it (a deferred action, or a button last frame).
        if (const std::string error = platform::takeDialogError(); !error.empty()) {
            wb().logLine(error);
            if (!d.unattended) message("File dialog", error);
        }
        d.refreshTitle();

        // Minimised with nothing else on screen: nothing to draw. Floating
        // windows (secondary viewports) stay on screen, and are drawn on.
        const bool iconified = glfwGetWindowAttrib(d.window, GLFW_ICONIFIED) != 0 && !d.unattended;
        if (iconified && ImGui::GetPlatformIO().Viewports.Size <= 1) {
            ImGui_ImplGlfw_Sleep(50);
            --d.frameDepth;
            return true;
        }

        d.updateViewports();
        d.drawFrame(false);
        ++d.frames;
        settings().autosave();
        --d.frameDepth;
        return !d.closing;
    }

    // One Dear ImGui frame, rendered and presented. From the refresh callback
    // (`fromRefresh`: inside glfwPollEvents, during Windows' modal size / move
    // loop) it only draws: GLFW is not polled from a callback, and what runs
    // between frames (posted functions, deferred actions), the screenshot and
    // the settings wait for the frame loop. The floating windows are updated
    // as in any frame, since Dear ImGui needs that every frame.
    void App::Impl::drawFrame(bool fromRefresh) {
        const bool iconified = glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0 && !unattended;
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // any input keeps the frames coming for a moment: hover states, popups
        // and tooltips settle over a few frames
        const ImGuiIO& io = ImGui::GetIO();
        if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f || io.MouseWheel != 0.0f || ImGui::IsAnyMouseDown() ||
            io.InputQueueCharacters.Size > 0 || ImGui::IsAnyItemActive())
            redrawFrames = std::max(redrawFrames, 3);
        for (ImGuiKey k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END; k = static_cast<ImGuiKey>(k + 1))
            if (ImGui::IsKeyDown(k)) {
                redrawFrames = std::max(redrawFrames, 3);
                break;
            }

        dialogHadFocus = dialogHasFocus;
        dialogHasFocus = false;
        claimedKeys.swap(claimingKeys);
        claimingKeys.clear();
        handleShortcuts();
        drawTitleBar();
        drawStatusBar();
        drawDockWindows();

        // dialogs: the modal ones as a stack, the others as windows
        for (std::size_t i = 0; i < dialogs.size(); ++i)
            if (!dialogs[i]->modal()) drawDialogs(i);
        {
            std::size_t first = 0;
            while (first < dialogs.size() && !dialogs[first]->modal()) ++first;
            drawDialogs(first);
        }
        {
            std::vector<std::shared_ptr<Dialog>> gone;
            for (auto it = dialogs.begin(); it != dialogs.end();) {
                if (!(*it)->isOpen()) {
                    gone.push_back(*it);
                    it = dialogs.erase(it);
                } else {
                    ++it;
                }
            }
            for (auto& g : gone) {
                // after the frame: the callback may open the next dialog
                self.defer([this, g] { g->closed(self); });
            }
        }

        ImGui::Render();
        if (!iconified) {
            int fbw = 0, fbh = 0;
            glfwGetFramebufferSize(window, &fbw, &fbh);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, fbw, fbh);
            const ImVec4 bg = theme::vec(theme::kBg);
            glClearColor(bg.x, bg.y, bg.z, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* current = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();   // skips the minimised main viewport
            glfwMakeContextCurrent(current);
        }
        if (iconified) {
            // the main window's vsynced swap paces the loop; minimised, nothing does
            ImGui_ImplGlfw_Sleep(16);
            return;
        }
        if (!fromRefresh) captureScreenshot();
        glfwSwapBuffers(window);
    }

    // An exception broke off a frame drawn from the refresh callback, where
    // it must not propagate. The frame is finished here instead: the windows
    // and stacks it left open are closed the way Dear ImGui's error recovery
    // does (reported, but not asserted on: the exception is the error), and
    // the frame is ended and its floating windows updated, so the next
    // NewFrame finds the state any finished frame leaves.
    void App::Impl::abandonFrame(const std::string& what) {
        wb().logLine("Error: " + what);
        ImGuiContext& g = *GImGui;
        ImGuiIO& io = ImGui::GetIO();
        if (g.WithinFrameScope) {
            const bool asserts = io.ConfigErrorRecoveryEnableAssert;
            io.ConfigErrorRecoveryEnableAssert = false;
            ImGui::ErrorRecoveryTryToRecoverState(&g.StackSizesInNewFrame);
            ImGui::EndFrame();
            io.ConfigErrorRecoveryEnableAssert = asserts;
        }
        if (g.FrameCountEnded == g.FrameCount && g.FrameCountPlatformEnded < g.FrameCount) ImGui::UpdatePlatformWindows();
        glfwMakeContextCurrent(window);   // a floating window's context may be the one left current
        redrawFrames = std::max(redrawFrames, 3);
    }

    bool App::screenshot(const std::string& path) {
        Impl& d = *impl_;
        // two frames: what the last action changed is laid out in the first
        // and drawn in the second
        requestRedraw(2);
        frame();
        d.pendingScreenshot = path;
        d.screenshotOk = false;
        requestRedraw(1);
        frame();
        return d.screenshotOk;
    }

    void App::quitNow() { impl_->finishClose(); }

    bool App::closing() const { return impl_->closing; }

    void App::requestClose() {
        Impl& d = *impl_;
        if (d.closeAsked) return;
        if (d.unattended) {
            // Nobody is at the keyboard: a question here would hold the
            // application open for ever. The run is cancelled and the window closes.
            d.finishClose();
            return;
        }
        auto afterUnsaved = [this] {
            Impl& d = *impl_;
            // A task (a dataset loading, an export) counts as much as a run.
            const bool run = bridge_.running(), task = bridge_.taskRunning();
            if (!run && !task) {
                // a worker job on a cluster is the user's to keep or cancel
                if (d.cluster) {
                    d.cluster->settleBeforeQuit([this] { impl_->finishClose(); });
                    if (!impl_->closing) impl_->closeAsked = false;
                    return;
                }
                d.finishClose();
                return;
            }
            const std::string what = run ? std::string("A run is in progress. Cancel it and quit?")
                                         : bridge_.taskLabel() + " is still in progress. Cancel it and quit?";
            ask("Quit", what, {"No", "Yes"}, [this](int answer) {
                impl_->closeAsked = false;
                if (answer == 1) impl_->finishClose();
            });
        };
        d.closeAsked = true;
        // A question needs the window on screen: closed from the taskbar while
        // minimised, it would otherwise wait unseen and hold every later close.
        if ((d.unsavedWork() || bridge_.busy()) && glfwGetWindowAttrib(d.window, GLFW_ICONIFIED)) glfwRestoreWindow(d.window);
        // Edits since the last save go with the window: say so first.
        if (d.unsavedWork()) {
            ask("Quit", "The pipeline has unsaved changes, and label edits are only kept by an export. Quit anyway?", {"Yes", "No"},
                [this, afterUnsaved](int answer) {
                    if (answer != 0) {
                        impl_->closeAsked = false;
                        return;
                    }
                    afterUnsaved();
                });
            return;
        }
        afterUnsaved();
        if (!d.closing && d.dialogs.empty()) d.closeAsked = false;
    }

    // --- dialogs and boxes ---------------------------------------------------------

    void App::showDialog(std::shared_ptr<Dialog> dialog) {
        if (!dialog) return;
        Impl& d = *impl_;
        if (std::find(d.dialogs.begin(), d.dialogs.end(), dialog) != d.dialogs.end()) {
            dialog->raise();
            return;
        }
        dialog->open_ = true;
        dialog->appeared_ = false;
        d.dialogs.push_back(std::move(dialog));
        requestRedraw();
    }

    bool App::dialogOpen() const {
        for (const auto& d : impl_->dialogs)
            if (d->modal()) return true;
        return false;
    }

    void App::message(const std::string& title, const std::string& text, MessageIcon) {
        if (impl_->unattended) {
            // nobody to close it; the text is in the log for whoever reads the run
            wb().logLine(title + ": " + simplified(text));
            return;
        }
        showDialog(std::make_shared<MessageBox>(title, text, std::vector<std::string>{"OK"}, nullptr));
    }

    void App::ask(const std::string& title, const std::string& text, const std::vector<std::string>& buttons,
                  std::function<void(int)> answer, int unattendedAnswer) {
        if (impl_->unattended) {
            if (answer) answer(unattendedAnswer);
            return;
        }
        showDialog(std::make_shared<MessageBox>(title, text, buttons, std::move(answer)));
    }

    void App::promptText(const std::string& title, const std::string& label, const std::string& initial,
                         std::function<void(const std::string&)> accepted, bool password) {
        showDialog(std::make_shared<TextPrompt>(title, label, initial, std::move(accepted), password));
    }

    // --- commands: datasets and pipelines --------------------------------------------

    std::string App::lastDir() const { return impl_->lastDir; }
    void App::setLastDir(const std::string& dir) { impl_->lastDir = dir; }

    std::vector<std::string> App::recentFiles() { return settings().getStringList("recent/datasets"); }

    void App::addRecentFile(const std::string& path) {
        // One entry per file, however its path was written (a drop, a dialog,
        // the command line, a list saved by an older version). A cluster path
        // is a name, not a path of this machine.
        const std::string key = isRemoteDatasetPath(path) ? path : absolutePath(path);
        auto same = [&key](const std::string& entry) {
            if (isRemoteDatasetPath(entry) || isRemoteDatasetPath(key)) return entry == key;
#ifdef _WIN32
            return toLower(absolutePath(entry)) == toLower(key);   // Windows paths ignore case
#else
            return absolutePath(entry) == key;
#endif
        };
        std::vector<std::string> list = recentFiles();
        list.erase(std::remove_if(list.begin(), list.end(), same), list.end());
        list.insert(list.begin(), key);
        if (list.size() > static_cast<std::size_t>(kMaxRecent)) list.resize(static_cast<std::size_t>(kMaxRecent));
        settings().set("recent/datasets", list);
    }

    void App::openDatasetDialog() {
        showDialog(makeOpenDatasetDialog(*this, wb().hasDataset() ? wb().dataset().sourcePath : std::string(),
                                         [this](const std::string& path, const OpenOptions& options) { openWith(path, options); }));
    }

    // `options.readAll` is the caller's: the Open dialog's "Read as", or a
    // full load where nothing was asked (a drop, a folder with a manifest,
    // the command line). A full load that would not fit is refused by
    // openDataset itself (fullLoadLimitBytes) and the dataset opens lazily.
    void App::openWith(const std::string& path, OpenOptions options) {
        if (bridge_.running() || !wb().canEdit()) {
            message("Open dataset", "A run is in progress: cancel it (Esc) or wait before opening a dataset.", MessageIcon::Info);
            return;
        }
        if (!bridge_.openDatasetAsync(path, std::move(options))) {
            message("Open dataset", "Another task is still running: cancel it or wait.");
            return;
        }
        addRecentFile(path);
        if (!isRemoteDatasetPath(path)) impl_->lastDir = parentPath(path);
    }

    // A folder with a manifest opens directly; otherwise the pattern dialog
    // builds one first.
    void App::openFolderDataset() {
        std::string start = impl_->lastDir;
        if (!start.empty() && isDirectory(start)) start = parentPath(start);
        const std::string folder = platform::pickFolderDialog("Open folder as dataset", start);
        if (folder.empty()) return;
        impl_->lastDir = folder;
        if (isFolderDataset(folder)) {
            OpenOptions o;
            o.readAll = true;
            openWith(folder, o);
            return;
        }
        showDialog(makeFolderDatasetDialog(*this, folder));
    }

    void App::openDatasetPath(const std::string& path) {
        if (isRemoteDatasetPath(path)) {
            OpenOptions o;
            o.readAll = false;   // it stays on the cluster: the viewer gets what it draws
            openWith(path, o);
            return;
        }
        if (isDirectory(path) && !isFolderDataset(path)) {
            bool store = false;
            for (const char* marker : {".zarray", ".zgroup", "zarr.json", "attributes.json"})
                if (pathExists(path + "/" + marker)) store = true;
            if (!store) {
                impl_->lastDir = path;
                showDialog(makeFolderDatasetDialog(*this, path));
                return;
            }
        }
        impl_->lastDir = parentPath(path);
        OpenOptions o;
        o.readAll = true;
        openWith(path, o);
    }

    void App::openPipelinePath(const std::string& path) {
        // Refused like every other edit while a run or a task holds the
        // pipeline: a dataset load finishing later would install its dataset
        // over this pipeline and reset its Load step. This also covers a load
        // that ended while the file dialog was open.
        if (bridge_.busy()) {
            message("Load pipeline",
                    (bridge_.running() ? std::string("A run") : bridge_.taskLabel()) +
                        " is still in progress: cancel it (Esc) or wait before loading a pipeline.",
                    MessageIcon::Info);
            return;
        }
        try {
            wb().loadPipeline(path);
            impl_->savedRevision = wb().history().revision();   // a loaded pipeline is a saved one
            impl_->lastDir = parentPath(path);
        } catch (const std::exception& e) {
            wb().logLine(std::string("Load pipeline failed: ") + e.what());
            message("Load pipeline", e.what());
        }
        requestRedraw();
    }

    void App::dropPaths(const std::vector<std::string>& dropped) {
        std::vector<std::string> paths;
        for (const std::string& p : dropped)
            if (!p.empty()) paths.push_back(isRemoteDatasetPath(p) ? p : absolutePath(p));
        if (paths.empty()) return;
        // a cluster dataset (the command line, a script): opened through the cluster session
        if (isRemoteDatasetPath(paths.front())) {
            if (!wb().canEdit() || bridge_.taskRunning()) {
                wb().logLine("Busy: cancel what runs (Esc) or wait before opening " + paths.front() + ".");
                return;
            }
            openDatasetPath(paths.front());
            return;
        }
        // A drop is an edit: the workbench refuses every one of them while a
        // run holds the pipeline.
        if (!wb().canEdit()) {
            wb().logLine("A run is in progress: cancel it (Esc) or wait before opening " + paths.front() + ".");
            return;
        }
        // A pipeline or a dataset also waits for a task (a dataset loading),
        // which would install its result over it; a .py operation does not.
        auto waitsForTask = [this](const std::string& path) {
            if (!bridge_.taskRunning()) return false;
            wb().logLine(bridge_.taskLabel() + " is in progress: cancel it (Esc) or wait before opening " + path + ".");
            return true;
        };
        // Several dataset files from one folder at once are what a folder
        // dataset is for, so offer that rather than opening one and dropping
        // the rest.
        const std::string folder = parentPath(paths.front());
        const bool manyDatasets = paths.size() > 1 && std::all_of(paths.begin(), paths.end(), [&folder](const std::string& p) {
                                      return kindOfDrop(p) == DropKind::Dataset && parentPath(p) == folder;
                                  });
        if (manyDatasets) {
            if (waitsForTask(folder)) return;
            wb().logLine("Dropped " + std::to_string(paths.size()) + " files: opening " + folder + " as a folder dataset.");
            openDatasetPath(folder);
            return;
        }
        for (const std::string& path : paths) {
            const DropKind kind = kindOfDrop(path);
            switch (kind) {
                case DropKind::Pipeline:
                    if (!waitsForTask(path)) openPipelinePath(path);
                    break;
                case DropKind::Plugin: pluginManager(path); break;
                case DropKind::Folder:
                case DropKind::Dataset:
                    if (!waitsForTask(path)) openDatasetPath(path);
                    break;
                case DropKind::None:
                    wb().logLine("Nothing to open in " + path +
                                 ": expected a TIFF / zarr / N5, a folder, a pipeline .toml or a .py operation.");
                    break;
            }
            // one dataset at a time: a second would replace the first
            if (kind == DropKind::Dataset || kind == DropKind::Folder) break;
        }
    }

    void App::savePipelineTo(const std::string& target) {
        std::string path = target;
        if (path.empty()) {
            path = platform::saveFileDialog("Save pipeline", impl_->lastDir, "pipeline.sirius.toml", {{"SIRIUS pipeline", "toml"}});
            if (path.empty()) return;
            if (!endsWithNoCase(path, ".toml")) path += ".sirius.toml";
        }
        try {
            wb().savePipeline(path);
            impl_->lastDir = parentPath(path);
            impl_->savedRevision = wb().history().revision();
        } catch (const std::exception& e) {
            message("Save pipeline", e.what());
        }
        requestRedraw();
    }

    void App::loadPipeline() {
        const std::string path = platform::openFileDialog("Load pipeline", impl_->lastDir, {{"SIRIUS pipeline", "toml"}});
        if (!path.empty()) openPipelinePath(path);
    }

    // Start or stop the session recording. The file is JSON lines, so it is
    // appended to and stays readable if the application is killed.
    void App::toggleRecording() {
        if (wb().recording()) {
            wb().stopRecording();
            return;
        }
        std::string path = platform::saveFileDialog("Record this session to", impl_->lastDir, "session.jsonl",
                                                    {{"Session recording", "jsonl"}});
        if (path.empty()) return;
        if (fileName(path).find('.') == std::string::npos) path += ".jsonl";
        try {
            wb().startRecording(path);
        } catch (const std::exception& e) {
            message("Record session", e.what());
        }
    }

    // --- commands: export ------------------------------------------------------------------

    void App::exportResultDialog() {
        if (!wb().hasDataset()) return;
        showDialog(makeExportDialog(*this, [this](int step, const ExportOptions& chosen) {
            // A run or a task may have started while the dialog was open (the
            // assistant, a drop): nothing, not even the sidecar, is written then.
            if (bridge_.busy()) {
                message("Export", "A run or task is in progress: cancel it (Esc) or wait, then export again.", MessageIcon::Info);
                return;
            }
            ExportOptions options = chosen;
            std::shared_ptr<const StepOutput> out = wb().output(step);
            if (!out) {
                message("Export", "Step " + Step::number(step) + " has not been computed yet. Run it first.", MessageIcon::Info);
                return;
            }
            const std::string pipelinePath = options.path + ".pipeline.toml";
            const bool sidecar = options.includePipeline;
            options.includePipeline = false;
            if (sidecar) {
                // Pipeline::save, not Workbench::savePipeline: that one makes
                // the file the pipeline's own, so Ctrl+S after an export would
                // overwrite <export>.pipeline.toml.
                try {
                    wb().pipeline().save(pipelinePath);
                    wb().logLine("Pipeline sidecar written to " + pipelinePath);
                } catch (const std::exception& e) {
                    wb().logLine(std::string("Pipeline sidecar: ") + e.what());
                }
            }
            // The labels are copied first: the task reads them on its thread
            // while the viewer may still paint into the step's volume.
            std::shared_ptr<const LabelVolume> labels = out->labels ? out->labels->clone() : nullptr;
            const bool started =
                bridge_.startTask("Export", [out, labels, options](const Bridge::TaskProgress& progress, const Bridge::TaskCancelled& cancelled) {
                    ArrayPtr array = out->asInput().materialize(progress);
                    exportArray(*array, out->meta, labels.get(), options, progress, cancelled);
                });
            if (!started) message("Export", "The export could not start: another task is running. Cancel it (Esc) or wait, then export again.");
        }));
    }

    void App::exportTrainingDialog() {
        if (!wb().hasDataset() || bridge_.busy()) return;
        showDialog(makeTrainingExportDialog(*this, [this](int step, const TrainingExportOptions& chosen) {
            if (bridge_.busy()) {
                message("Export training data", "A run or task is in progress: cancel it (Esc) or wait, then export again.",
                        MessageIcon::Info);
                return;
            }
            std::shared_ptr<const StepOutput> out = wb().output(step);
            if (!out || !out->labels || out->labels->empty()) {
                message("Export training data", "Step " + Step::number(step) + " has no labels. Run a segmentation step first.",
                        MessageIcon::Info);
                return;
            }
            TrainingExportOptions options = chosen;
            // the pipeline goes with the sample: it is how the labels were made
            options.provenance = {{"step", Step::number(step)},
                                  {"step_name", wb().pipeline().at(step).name},
                                  {"kind", wb().pipeline().at(step).kind},
                                  {"dataset", wb().hasDataset() ? wb().dataset().sourcePath : std::string()},
                                  {"pipeline", wb().pipeline().toJson()}};
            // a copy: the task reads on its thread while the viewer may paint
            std::shared_ptr<const LabelVolume> labels = out->labels->clone();
            const bool started = bridge_.startTask(
                "Export training data", [out, labels, options](const Bridge::TaskProgress& progress, const Bridge::TaskCancelled& cancelled) {
                    ArrayPtr array = options.image || options.slices ? out->asInput().materialize(progress) : nullptr;
                    const Array5 empty;
                    exportTrainingData(array ? *array : empty, out->meta, *labels, options, progress, cancelled);
                });
            if (!started)
                message("Export training data", "The export could not start: another task is running. Cancel it (Esc) or wait, then export again.");
        }));
    }

    void App::exportPythonScript() {
        const std::string path = platform::saveFileDialog("Export pipeline as Python", impl_->lastDir, "pipeline.py", {{"Python", "py"}});
        if (path.empty()) return;
        const std::string script = wb().pipeline().toPythonScript(wb().hasDataset() ? wb().dataset().sourcePath : std::string());
        std::ofstream f(std::filesystem::u8path(path), std::ios::binary);
        if (!f) {
            message("Export", "Cannot write " + path);
            return;
        }
        f << script;
        wb().logLine("Python script written to " + path);
    }

    void App::exportFigureImage() {
        std::vector<std::uint8_t> rgba;
        int w = 0, h = 0;
        if (!impl_->viewer->grabView(rgba, w, h)) {
            // The slices are grabbed as the last frame drew them, which it
            // does not while the viewer is behind a tab or under the cover.
            wb().logLine(impl_->viewerDrawn ? "Export figure: the view could not be grabbed."
                                            : "Export figure: the viewer is not on screen (behind another tab, or under a maximised panel).");
            return;
        }
        std::string path = platform::saveFileDialog("Export figure", impl_->lastDir, "figure.png", {{"PNG", "png"}});
        if (path.empty()) return;
        if (!endsWithNoCase(path, ".png")) path += ".png";
        if (!writePng(path, rgba.data(), w, h)) message("Export figure", "Cannot write " + path);
        else wb().logLine("Figure written to " + path);
    }

    void App::exportLabels() {
        std::shared_ptr<LabelVolume> labelsVol = wb().viewedLabels();
        if (!labelsVol || labelsVol->empty()) {
            wb().logLine("Export labels: the viewed step has no labels.");
            return;
        }
        std::string path = platform::saveFileDialog("Export labels", impl_->lastDir, "labels.tif", {{"TIFF", "tif,tiff"}});
        if (path.empty()) return;
        if (!endsWithNoCase(path, ".tif") && !endsWithNoCase(path, ".tiff")) path += ".tif";
        try {
            writeTiffStack<std::uint32_t>(path, labelsVol->view().asStack(), TiffCompression::Deflate);
            wb().logLine("Labels written to " + path);
        } catch (const std::exception& e) {
            message("Export labels", e.what());
        }
    }

    // --- commands: steps --------------------------------------------------------------------

    // "Run selected step" against "Run to selected step". The executor only
    // ever walks the pipeline from the top, taking each step's fresh cache
    // entry as it passes, so the one thing that separates the two is whether
    // the steps above are already computed: "Run to" recomputes whatever is
    // stale, "Run selected" is the step on its own and refuses -- out loud --
    // when its input is missing.
    void App::runSelectedStep() {
        const int i = wb().selectedIndex();
        const Pipeline& p = wb().pipeline();
        if (i < 0 || i >= p.size()) return;
        for (int j = 0; j < i; ++j) {
            if (j > 0 && !p.at(j).enabled) continue;
            if (wb().outputFresh(j)) continue;
            wb().logLine("Run selected step: step " + Step::number(j) + " " + p.at(j).name +
                         " above it is not computed yet, so this step has no input. "
                         "Use Process \xE2\x96\xB8 Run to selected step (or Run all enabled) instead.");
            return;
        }
        bridge_.startRun(i);
    }

    void App::runAll() { bridge_.startRun(-1); }

    void App::runTo(int index) { bridge_.startRun(index); }

    void App::cancel() {
        if (bridge_.running()) bridge_.cancelRun();
        if (bridge_.taskRunning()) bridge_.cancelTask();
    }

    // Backspace removes a step with no dialog in the way, which is right: the
    // removal is one undo entry. The log and the status bar name the step and
    // point at Undo. A step holding a computed output is the one case worth a
    // question: undo brings the step back but not its cache.
    void App::removeStepAt(int index) {
        const Pipeline& p = wb().pipeline();
        if (index <= 0 || index >= p.size() || !wb().canEdit()) return;
        const Step& step = p.at(index);
        const StepId id = step.id;
        const std::string name = Step::number(index) + " " + step.name;
        const std::size_t cached = wb().executor().cachedBytesOf(step.id);
        auto remove = [this, id, name] {
            // the step asked about is found again by id: the assistant may
            // have moved or removed steps meanwhile
            const int now = wb().pipeline().indexOf(id);
            if (now <= 0) return;
            wb().removeStep(now);   // logs "Removed step ..." itself
            wb().logLine("Edit \xE2\x96\xB8 Undo (" + shortcutText(keys::undo) + ") brings step " + name + " back.");
        };
        if (cached > 64ull * 1024 * 1024) {
            ask("Remove step", "Remove step " + name + "?\n\nIts computed output (" + bytesText(cached) + ") is discarded and has to be recomputed if you bring the step back.", {"Cancel", "Remove"}, [remove](int answer) {
                    if (answer == 1) remove(); }, 1);
            return;
        }
        remove();
    }

    int App::segmentationStep() const {
        const Pipeline& p = bridge_.wb().pipeline();
        for (int i = p.size() - 1; i >= 0; --i)
            if (p.at(i).op().info().producesLabels) return i;
        return -1;
    }

    // The segmentation step the model goes to: the selected one, else the
    // first, else a new one.
    int App::segmentationStepOrNew() {
        int i = segmentationStep();
        if (i < 0 || wb().pipeline().at(i).kind != "seg") {
            if (!findOperation("seg")) return -1;
            i = wb().pipeline().indexOf(wb().addStep("seg"));
        }
        return i;
    }

    // The last step of this kind, or a new one at the end.
    int App::stepOrNew(const std::string& kind) {
        const Pipeline& p = wb().pipeline();
        for (int i = p.size() - 1; i >= 0; --i)
            if (p.at(i).kind == kind) return i;
        if (!findOperation(kind)) return -1;
        return p.indexOf(wb().addStep(kind));
    }

    void App::modelHub() {
        // On the tab that matches the step in hand: with a foundation step
        // selected the menu means "which bundle".
        const int selected = wb().selectedIndex();
        const bool bundles = selected >= 0 && selected < wb().pipeline().size() && wb().pipeline().at(selected).kind == "foundation";
        showDialog(makeModelHubDialog(*this, bundles, [this](const std::string& chosen) {
            if (chosen.empty()) return;
            // A bundle is the foundation step's model, not the segmentation step's.
            const int i = endsWithNoCase(chosen, ".ltb") ? stepOrNew("foundation") : segmentationStepOrNew();
            if (i < 0) return;
            wb().setStepParam(i, "model", chosen);
            wb().select(i);
        }));
    }

    void App::loadTorchModel() {
        // -1 when no step could be added: the workbench refuses every edit during a run.
        const int i = segmentationStepOrNew();
        if (i < 0) return;
        const Step& s = wb().pipeline().at(i);
        const StepId id = s.id;
        std::string pathKey;
        for (const ParamSpec& spec : s.op().info().params)
            if (spec.type == ParamType::Path) {
                pathKey = spec.key;
                break;
            }
        if (pathKey.empty()) return;
        const std::string path = platform::openFileDialog("Load Torch model", impl_->lastDir, {{"TorchScript / ONNX", "pt,pth,ts,onnx"}});
        const int now = wb().pipeline().indexOf(id);
        if (path.empty() || now < 0) return;
        wb().setStepParam(now, pathKey, path);
        wb().select(now);
    }

    void App::mergeLabelsDialog() {
        promptText("Merge labels", "Label ids to merge (comma-separated)", std::to_string(wb().viewState().selectedLabel),
                   [this](const std::string& text) {
                       std::vector<std::uint32_t> ids;
                       for (const std::string& part : split(text, ',', true)) {
                           try {
                               const long long v = std::stoll(trimmed(part));
                               if (v > 0) ids.push_back(static_cast<std::uint32_t>(v));
                           } catch (const std::exception&) {
                           }
                       }
                       if (ids.size() >= 2) wb().mergeLabels(ids);
                       else wb().logLine("Merge labels: give at least two label ids.");
                   });
    }

    void App::selectFlagged(bool forward) {
        const std::uint32_t id = wb().nextFlaggedLabel(forward);
        if (!id) {
            wb().logLine("No flagged labels.");
            return;
        }
        ViewState s = wb().viewState();
        s.selectedLabel = id;
        s.labels = true;
        wb().setViewState(s);
    }

    // --- commands: windows ------------------------------------------------------------------

    void App::preferences() { showDialog(makePreferencesDialog(*this)); }

    void App::clusterDialog() {
        // one at a time: a second Connect would only show the same session twice
        for (const auto& d : impl_->dialogs)
            if (d && d->isOpen() && d->title() == "Connect to cluster") {
                d->raise();
                return;
            }
        showDialog(makeClusterDialog(*this));
    }

    ClusterLink& App::cluster() {
        if (!impl_->cluster) impl_->cluster = std::make_unique<ClusterLink>(*this);
        return *impl_->cluster;
    }

    void App::pluginManager(const std::string& file) {
        Impl& d = *impl_;
        if (!d.plugins) d.plugins = makePluginManager(*this);
        if (!d.plugins) return;
        if (!file.empty()) d.plugins->openFile(file);
        showDialog(d.plugins);
    }

    void App::showHelpForSelected() {
        const int i = wb().selectedIndex();
        if (i >= 0 && i < wb().pipeline().size()) impl_->help->showKind(wb().pipeline().at(i).kind);
        else impl_->help->setVisible(true);
    }

    void App::showHelp(const std::string& kind) { impl_->help->showKind(kind); }

    void App::toggleHelp() {
        if (impl_->help->visible()) impl_->help->setVisible(false);
        else showHelpForSelected();
    }

    bool App::helpOpen() const { return impl_->help && impl_->help->visible(); }

    void App::showLog() {
        impl_->showLog = true;
        impl_->focusLog = true;
        requestRedraw();
    }

    void App::showOperations() {
        impl_->showOps = true;
        impl_->focusOps = true;
        requestRedraw();
    }

    void App::setAssistantVisible(bool on) {
        impl_->showAssistant = on;
        impl_->focusAssistant = on;
        requestRedraw();
    }

    bool App::assistantVisible() const { return impl_->showAssistant; }

    void App::askAssistant(const std::string& text) {
        setAssistantVisible(true);
        impl_->assistant->ask(text);
    }

    void App::setDiagnosticsMaximized(bool on) {
        // The cover over the viewer is the one maximised panel. A maximised
        // viewer goes back into its dock, where the cover finds it.
        const bool viewerComesBack = on && impl_->placeOf(Panel::Viewer).maximized;
        if (on) impl_->restoreMaximized(Panel::Diag);
        impl_->diagMaximized = on;
        if (on) impl_->showDiag = true;
        // otherwise nothing on screen would say why nothing changed
        if (on && !impl_->viewerInFront && !viewerComesBack)
            wb().logLine("Diagnostics maximised: they cover the viewer once it is docked in the main window with its tab in front.");
        requestRedraw();
    }

    bool App::diagnosticsMaximized() const { return impl_->diagMaximized; }

    void App::floatDiagnostics() {
        impl_->showDiag = true;
        impl_->diagMaximized = false;
        // Floating already (under the cover, say), it stays where it floats.
        if (!impl_->placeOf(Panel::Diag).floating) impl_->placeOf(Panel::Diag).request = Move::Float;
        requestRedraw();
    }

    bool App::diagnosticsFloating() const { return impl_->placeOf(Panel::Diag).floating; }

    void App::dockDiagnostics() {
        impl_->showDiag = true;
        impl_->diagMaximized = false;
        // Only the diagnostics move (drawDockWindows finds them a dock): the
        // rest of the arrangement stays as the user left it.
        if (impl_->placeOf(Panel::Diag).floating) impl_->placeOf(Panel::Diag).request = Move::Dock;
        requestRedraw();
    }

    void App::resetLayout() {
        Impl& d = *impl_;
        d.rebuildLayout = true;
        d.showOps = d.showParams = d.showDiag = d.showLog = true;
        d.diagMaximized = false;
        for (Impl::Place& p : d.places) {
            p.maximized = false;
            p.request = Move::None;
        }
        wb().logLine("Layout reset to the default arrangement.");
        requestRedraw();
    }

    void App::about() {
        message("About SIRIUS",
                std::string("SIRIUS \xE2\x80\x94 Structured Illumination Reconstruction and Image Utility Suite\n"
                            "Microscopy processing workbench.\n\n"
                            "CPU, CUDA and HPC backends \xC2\xB7 TIFF, OME-TIFF") +
                    (zarrSupported() ? ", zarr / N5 (TensorStore)" : "") + " \xC2\xB7 Dear ImGui " + IMGUI_VERSION + "\n" + kRepositoryUrl,
                MessageIcon::Info);
    }

    // --- panels --------------------------------------------------------------------------------

    Viewer& App::viewer() { return *impl_->viewer; }
    OpsPanel& App::ops() { return *impl_->ops; }
    ParamsPanel& App::params() { return *impl_->params; }
    DiagnosticsPanel& App::diagnostics() { return *impl_->diagnostics; }
    AssistantPanel& App::assistant() { return *impl_->assistant; }
    LogPanel& App::log() { return *impl_->log; }
    HelpWindow& App::help() { return *impl_->help; }

    // --- scripting -------------------------------------------------------------------------------

    bool App::triggerAction(const std::string& text) {
        static const std::string dots = "\xE2\x80\xA6";
        for (Action& a : impl_->actions) {
            if (!a.run) continue;
            const std::string label = a.label ? a.label() : a.text;
            for (const std::string& candidate : {a.text, label}) {
                if (candidate == text || candidate == text + dots || startsWith(candidate, text + dots)) {
                    a.run();
                    return true;
                }
            }
        }
        wb().logLine("no action named " + text);
        return false;
    }

} // namespace sirius::app::gui
