// sirius-cli: the SIRIUS workbench without a window, for scripts and agents
// (app/cli/README.md).
//
//   sirius-cli [global options] <command> [options]
//
// One-shot commands print one JSON document on stdout; `session` speaks JSON
// lines and `mcp` the Model Context Protocol on stdin / stdout. Logs and
// progress go to stderr. The process ends through exitProcess(): a detached
// stdin reader may still be blocked when the work is done, and static
// destructors must not run under it.

#include <exception>
#include <string>
#include <vector>

#include "cli/commands.hpp"
#include "cli/stdio.hpp"

int main(int argc, char** argv) {
    // First: nothing may reach the real stdout but our own documents.
    sirius::cli::takeOverStdout();
    // UTF-8 on Windows too: the executable's manifest sets the process code page.
    const std::vector<std::string> args(argv + 1, argv + argc);
    int code = 1;
    try {
        code = sirius::cli::run(args);
    } catch (const std::exception& e) {
        sirius::cli::writeError(std::string("sirius-cli: internal: ") + e.what() + "\n");
    } catch (...) {
        sirius::cli::writeError("sirius-cli: internal: unknown error\n");
    }
    sirius::cli::exitProcess(code);
}
