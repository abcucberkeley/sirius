#include "imgui/worker_launcher.hpp"

#include "imgui/settings.hpp"
#include "imgui/strings.hpp"

namespace sirius::app::gui {

    WorkerLauncher::WorkerLauncher() {
        // Read at every start, so the next worker follows a change made in
        // Preferences; $SIRIUS_PYTHON still comes before the setting, and
        // SIRIUS's own environment after it (pyenv::workerInterpreter).
        setConfiguredPython([] { return trimmed(settings().getString("worker/python")); });
        setConfiguredScriptDir([] { return settings().getString("worker/dir"); });
        // The model hub installs model packages into the worker's interpreter.
        setAllowInstall(true);
        setSetupHint("Set up SIRIUS's own Python environment: Preferences \xE2\x96\xB8 Compute \xE2\x96\xB8 Python environment.");
    }

} // namespace sirius::app::gui
