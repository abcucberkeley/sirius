#ifndef SIRIUS_IMGUI_SETTINGS_HPP
#define SIRIUS_IMGUI_SETTINGS_HPP

// The application's persistent settings (core/settings_store.hpp): the GUI's
// spelling of them.

#include "core/settings_store.hpp"

namespace sirius::app::gui {

    using sirius::app::Settings;

    inline Settings& settings() { return Settings::instance(); }

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_SETTINGS_HPP
