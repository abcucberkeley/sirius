#ifndef SIRIUS_IMGUI_WORKER_LAUNCHER_HPP
#define SIRIUS_IMGUI_WORKER_LAUNCHER_HPP

// The GUI's launcher of the bundled Python worker: the core's LocalWorker
// (core/local_worker.hpp, which sirius-cli shares) with what the application
// adds to it -- the interpreter and the worker directory from the settings,
// installs for the model hub, and the Preferences hint when the worker cannot
// start. Installed into the workbench through Workbench::setLocalWorkerLauncher;
// the threading rules are LocalWorker's.

#include "core/local_worker.hpp"

namespace sirius::app::gui {

    class WorkerLauncher : public LocalWorker {
    public:
        WorkerLauncher();
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_WORKER_LAUNCHER_HPP
