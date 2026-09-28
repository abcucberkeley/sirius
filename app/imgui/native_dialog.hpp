#ifndef SIRIUS_IMGUI_NATIVE_DIALOG_HPP
#define SIRIUS_IMGUI_NATIVE_DIALOG_HPP

// The window side of the native file dialogs of platform.cpp: the parent they
// are shown over, and the input the application's windows received while one
// was open. Apart from platform.cpp because it needs the window system's
// headers, and X11's define None, Bool, Status and the like as macros.

#include <nfd.h>

namespace sirius::app::gui::platform::native_dialog {

    // The application window a dialog is opened from, as NFD's parent handle:
    // within a frame the platform window of the Dear ImGui window being drawn,
    // otherwise the main window (the one whose OpenGL context is current).
    // Unset (no parent) when that window is minimised, without a Dear ImGui
    // context, and on Wayland, where NFD takes none.
    nfdwindowhandle_t parentWindow();

    // After a blocking dialog: drops the clicks, wheel turns, keys and text
    // that were queued for the application's windows while it was open, so
    // they are not replayed into the next frame as a click on whatever lies
    // under the pointer, or as a shortcut. Pointer moves and focus changes are
    // kept. Nothing without a Dear ImGui context. `poll`: first read what the
    // window system still holds for the windows (X11, Wayland, macOS). False
    // inside GLFW's own event processing, which must not be polled from its
    // callbacks: what it has not delivered yet then reaches the next frame.
    void dropQueuedInput(bool poll);

} // namespace sirius::app::gui::platform::native_dialog

#endif // SIRIUS_IMGUI_NATIVE_DIALOG_HPP
