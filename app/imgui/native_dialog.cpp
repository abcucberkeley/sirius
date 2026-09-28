#include "imgui/native_dialog.hpp"

#include <imgui.h>
#include <imgui_internal.h>

// Last: glfw3native.h brings in the window system's headers (windows.h, Xlib.h).
// GLFW_BUILD_X11 builds decide SIRIUS_GLFW_X11 (app/imgui/CMakeLists.txt); a
// Wayland-only GLFW has no glfwGetX11Window to call.
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#elif defined(SIRIUS_GLFW_X11)
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <nfd_glfw3.h>

namespace sirius::app::gui::platform::native_dialog {

    nfdwindowhandle_t parentWindow() {
        nfdwindowhandle_t handle{};
        // Without a Dear ImGui context there is no window (GLFW may not even
        // be initialised).
        ImGuiContext* context = ImGui::GetCurrentContext();
        if (context == nullptr) return handle;
        // Within a frame, the platform window of the Dear ImGui window being
        // drawn: a floating viewport's when the request came from one, which
        // stays on screen while the main window is minimised. Otherwise (and
        // before a new viewport's window exists) the main window, whose
        // context is current both between frames and while a frame is built:
        // a viewport's is only made current while it is rendered, and the
        // main one is restored after.
        GLFWwindow* window = nullptr;
        if (context->WithinFrameScope && context->CurrentWindow != nullptr && context->CurrentWindow->Viewport != nullptr)
            window = static_cast<GLFWwindow*>(context->CurrentWindow->Viewport->PlatformHandle);
        if (window == nullptr) window = glfwGetCurrentContext();
        // No owner rather than a minimised one: Windows centres a dialog on its
        // owner, and a minimised window lies far off screen.
        if (window == nullptr || glfwGetWindowAttrib(window, GLFW_ICONIFIED)) return handle;
        if (!NFD_GetNativeWindowFromGLFWWindow(window, &handle)) handle = nfdwindowhandle_t{};
        return handle;
    }

    void dropQueuedInput(bool poll) {
        ImGuiContext* context = ImGui::GetCurrentContext();
        if (context == nullptr) return;
#ifndef _WIN32
        // X11 and Wayland keep what arrived for the windows in the connection
        // until GLFW reads it; on Windows the dialog's own message loop has
        // already dispatched it to GLFW, and so to the Dear ImGui backend.
        if (poll) glfwPollEvents();
#else
        (void)poll;
#endif
        ImVector<ImGuiInputEvent>& queue = context->InputEventsQueue;
        int kept = 0;
        for (int i = 0; i < queue.Size; ++i) {
            const ImGuiInputEventType type = queue[i].Type;
            if (type == ImGuiInputEventType_MouseButton || type == ImGuiInputEventType_MouseWheel ||
                type == ImGuiInputEventType_Key || type == ImGuiInputEventType_Text)
                continue;
            queue[kept++] = queue[i];
        }
        queue.resize(kept);
        // A key or button that went down before the dialog opened lost its
        // release with the events above. The pointer stays where it was, for
        // the hover under it until it moves.
        const ImVec2 pointer = context->IO.MousePos;
        context->IO.ClearInputKeys();
        context->IO.ClearInputMouse();
        context->IO.MousePos = pointer;
    }

} // namespace sirius::app::gui::platform::native_dialog
