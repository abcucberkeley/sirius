#ifndef SIRIUS_CLI_COMMANDS_HPP
#define SIRIUS_CLI_COMMANDS_HPP

// sirius-cli's commands and the JSON envelope each one-shot command prints on
// stdout:
//
//   {"ok":true,"schema":"sirius-cli/1","command":"run","result":{...},"warnings":[...]}
//   {"ok":false,"schema":"sirius-cli/1","command":"run","error":{"code","message","hint","data"},
//    "exit_code":4,"warnings":[...]}
//
// A failure is followed by one summary line on stderr ("sirius-cli: <code>: <message>").
// Every behaviour lives in the core's HeadlessWorkbench (core/headless.hpp); a
// command is a sequence of its tool calls, so the command line, the session and
// the MCP server cannot drift apart.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sirius::cli {

    constexpr const char* kOutputSchema = "sirius-cli/1";
    constexpr const char* kSessionProtocol = "sirius-session/1";

    // Runs one command line (argv without the program name); the exit code.
    // Cleans up after itself (the worker, the scratch directory) but leaves
    // ending the process to the caller (exitProcess).
    int run(const std::vector<std::string>& argv);

    struct Args;
    // `sirius-cli serve` (serve.cpp): listens, prints the announce line,
    // serves until shutdown; the exit code.
    int serveEngine(const Args& args);

    // The MCP protocol versions the server speaks, newest first.
    const std::vector<std::string>& mcpVersions();

    // --- usage.cpp ------------------------------------------------------------

    // The top-level --help text.
    std::string usageText();
    // `sirius-cli <command> --help`: usage, options, an output sample, exit
    // codes and examples; "worker" gives the worker commands' overview.
    std::string commandHelp(const std::string& command);
    // `sirius-cli schema`: the commands, their options, the exit codes, the
    // error codes with their exit codes and the envelope.
    nlohmann::json schemaJson();
    // What the MCP server tells a client in initialize (at most 2048 characters).
    std::string mcpInstructions();
    // Every error code sirius-cli reports, for schema; agent::exitCodeFor gives
    // each one's exit code.
    const std::vector<std::string>& errorCodes();

} // namespace sirius::cli

#endif // SIRIUS_CLI_COMMANDS_HPP
