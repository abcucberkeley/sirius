#ifndef SIRIUS_CLI_STDIO_HPP
#define SIRIUS_CLI_STDIO_HPP

// sirius-cli's standard streams: stdout carries only results and protocol
// messages (one line each), everything else goes to stderr; signals and the
// console's close events end the process in an orderly way.
//
// takeOverStdout() keeps the real stdout for writeLine() and points file
// descriptor 1 (and on Windows STD_OUTPUT_HANDLE) at stderr, so whatever a
// library prints lands on stderr instead of in the middle of a JSON document.

#include <functional>
#include <string>

namespace sirius::cli {
    void takeOverStdout();                              // first thing in main
    void writeLine(const std::string& line);            // thread-safe, to the kept stdout
    void writeRaw(const std::string& text);             // help --markdown
    bool stdinIsTerminal();
    bool stdoutIsTerminal();
    bool stderrIsTerminal();
    void installInterruptHandler(std::function<void()> onFirst);      // SIGINT / Ctrl+C; a second one: exitProcess(130)
    void installTerminationHandler(std::function<void()> onTerminate);// SIGTERM, SIGHUP, CTRL_CLOSE/BREAK/LOGOFF/SHUTDOWN
    void startStdinReader(std::function<void(std::string)> onLine, std::function<void()> onEof);   // detached thread
    [[noreturn]] void exitProcess(int code);            // emergency clean-up, flush, restore the code page, std::_Exit

    // Thread-safe, as written (no newline added); every stderr line of the
    // CLI goes through here so lines from different threads never interleave.
    void writeError(const std::string& text);
    // True once a write to the kept stdout failed: the reader of our output
    // is gone (a closed pipe), so a server can stop.
    bool outputClosed();
    // What exitProcess runs first (a second Ctrl+C, the timeout's watchdog):
    // the scratch directory is removed even when the main thread is stuck.
    void setEmergencyCleanup(std::function<void()> cleanup);
    // Everything on stdin up to its end (`--steps -`); false when it cannot be read.
    bool readStandardInput(std::string& out);
    // One answer typed on a terminal, up to Enter; false at the end of input.
    bool readAnswer(std::string& out);
} // namespace sirius::cli

#endif // SIRIUS_CLI_STDIO_HPP
