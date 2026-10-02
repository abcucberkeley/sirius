#ifndef SIRIUS_APP_REMOTE_HOST_HPP
#define SIRIUS_APP_REMOTE_HOST_HPP

// One SSH connection to a cluster's login node, opened once per session with
// the system's OpenSSH client, which does two jobs for the application:
//
//   * a command channel: the remote command is a login shell reading its
//     script from stdin (`bash -l -s`); run() writes one framed command at a
//     time and reads its stdout, its stderr and its exit code back;
//   * a SOCKS5 proxy (`-D 127.0.0.1:<port>`): the application reaches the
//     worker on a compute node through it (rpc::connectSocks5), the node's
//     name resolved on the cluster. It listens on 127.0.0.1 only and lives
//     exactly as long as the session (ssh is in a job object / its own
//     process group and ends with it); while it lives, any process on this
//     machine can open connections into the cluster through it as this user
//     (app/python/SECURITY.md). Forwarding the worker's port alone would need
//     the port before the job runs, or a second login.
//
// ssh runs with -x -a, ForwardAgent=no, ForwardX11=no and
// PermitLocalCommand=no, whatever ~/.ssh/config says: the cluster gets
// neither this machine's display nor its SSH agent.
//
// Logging in: ssh asks for the password, the one-time code, a host key
// confirmation through SSH_ASKPASS with SSH_ASKPASS_REQUIRE=force, which
// OpenSSH (Windows' 9.5 included) honours without a DISPLAY. The askpass
// program is this application's own executable (askpassMain, recognised by
// the environment it is started with): it hands the prompt to the running
// application over a loopback socket guarded by a per-session secret
// (AskpassServer) and prints the answer it gets. The answer is never kept:
// not in the settings, the secret store or a log.
//
// A wrong password costs one attempt (NumberOfPasswordPrompts=1); a login
// that fails is never retried here. A prompt the user cancels stops ssh
// before the helper returns, so ssh never sends an empty answer.
//
// Command framing (see run()): every command gets random begin / end
// markers; its script reaches the remote shell inside a quoted here-document
// on stdin (never in an argument list, so a secret in it is not visible to
// `ps`) and runs with `eval` in a subshell whose stdin is /dev/null, so a
// syntax error or an `exit` in it cannot end the session.
//
// GUI-free: sirius-cli can use it as the window does.

#include <atomic>
#include <stdexcept>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/process.hpp"

namespace sirius::app::ssh {

    // A login or a command that failed; what() is for the user, `detail` the
    // server's own words (ssh's stderr, the remote command's stderr).
    struct SshError : std::runtime_error {
        SshError(const std::string& what, std::string detail = {}) : std::runtime_error(what), detail(std::move(detail)) {}
        std::string detail;
    };

    // The OpenSSH client: on Windows %SystemRoot%\System32\OpenSSH\ssh.exe
    // first, then `ssh` on PATH; "" when there is none.
    std::string findSsh();
    // `s` single-quoted for a POSIX shell.
    std::string shellQuote(const std::string& s);
    // A remote path as a shell word: "~/x y" becomes "$HOME"/'x y', so that
    // the home directory is expanded and the rest is not.
    std::string remotePathWord(const std::string& path);
    // A random lowercase hex string of `bytes` bytes (the OS's generator).
    std::string randomHex(int bytes);

    // --- askpass ------------------------------------------------------------------

    // What ssh asked: its prompt and whether the answer may be shown (a
    // yes/no question) or is a secret.
    struct Prompt {
        std::string text;
        bool echo = false;        // OpenSSH's own host key confirmation: not a secret
        bool notifyOnly = false;  // SSH_ASKPASS_PROMPT=none: show, nothing to answer
    };

    // Whether a prompt is OpenSSH's own host key question, whose answer may
    // be shown: SSH_ASKPASS_PROMPT=confirm (`kind`), or OpenSSH's exact
    // wording ("The authenticity of host ... Are you sure you want to
    // continue connecting (yes/no..."). A server's keyboard-interactive
    // prompt never is, whatever it says.
    bool isHostKeyConfirmation(const std::string& kind, const std::string& text);

    // The application's end of the askpass relay: a loopback listener that
    // takes one prompt per connection from askpassMain and answers it with
    // what `handler` returns (on that connection's thread, one prompt at a
    // time, and may block until the user answers); nullopt = cancelled. A
    // connection that has not sent its line within a second is dropped, and
    // at most 8 are served at once: any local process can connect.
    class AskpassServer {
    public:
        using Handler = std::function<std::optional<std::string>(const Prompt& prompt)>;
        explicit AskpassServer(Handler handler);
        ~AskpassServer();
        AskpassServer(const AskpassServer&) = delete;
        AskpassServer& operator=(const AskpassServer&) = delete;

        int port() const noexcept { return port_; }
        // Stops listening: once the login is over nothing is asked any more,
        // and no process on this machine should find the port open. A
        // connection already in the handler is answered; the destructor
        // waits for it.
        void close();
        // The environment ssh gets: SSH_ASKPASS = `program`,
        // SSH_ASKPASS_REQUIRE=force, the port and the secret for the helper.
        std::vector<std::pair<std::string, std::string>> environment(const std::string& program) const;

    private:
        void serve();
        void serveOne(std::intptr_t connection);
        Handler handler_;
        std::string secret_;
        std::intptr_t listener_ = -1;
        int port_ = 0;
        std::atomic<bool> stop_{false};
        std::thread thread_;
        std::mutex handlerMutex_;
        std::mutex connMutex_;
        std::vector<std::thread> connections_;      // under connMutex_
        std::set<std::thread::id> doneIds_;          // finished connections, under connMutex_
    };

    // The helper's side, for main(): true when this process was started as
    // ssh's askpass program (the environment AskpassServer sets), and then
    // `exitCode` is what main returns. It prints the answer on stdout.
    bool isAskpassInvocation();
    int askpassMain(int argc, char** argv);

    // --- the session --------------------------------------------------------------

    struct Options {
        std::string program;                     // "" = findSsh()
        std::vector<std::string> programArgs;    // before ssh's own (tests: a fake ssh run by an interpreter)
        std::string host;                        // ssh destination: an alias of ~/.ssh/config, user@host
        int socksPort = -1;                      // -1: a free one; 0: no proxy
        std::string remoteCommand = "/bin/bash -l -s";
        std::vector<std::pair<std::string, std::string>> environment;   // added to ssh's (the askpass relay)
        std::vector<std::string> extraOptions;   // more "-o" values
    };

    // The argument list ssh is started with (exposed for tests).
    std::vector<std::string> sshArguments(const Options& options, int socksPort);

    struct CommandResult {
        int exitCode = -1;
        std::string out;
        std::string err;
        bool ok() const noexcept { return exitCode == 0; }
    };

    class Session {
    public:
        Session();
        ~Session();   // close()
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;

        // Starts ssh and waits until the remote shell has answered its first
        // command, which is when the login (prompts included) is over. Throws
        // SshError when ssh ends first (with what it printed), when
        // `cancelled` says to stop, or after `timeout`.
        void open(const Options& options, const std::function<bool()>& cancelled = {},
                  std::chrono::milliseconds timeout = std::chrono::minutes(5));
        bool isOpen();
        int socksPort() const noexcept { return socksPort_; }
        const std::string& host() const noexcept { return host_; }

        // One command (a bash script), serialised with every other caller.
        // Throws SshError when the session ends or `timeout` passes (the
        // command may then still be running remotely; its late output is
        // skipped by the next command's markers).
        CommandResult run(const std::string& script, std::chrono::milliseconds timeout = std::chrono::seconds(60),
                          const std::function<bool()>& cancelled = {});
        // Ends ssh (and with it the proxy). Safe to call at any time.
        void close();
        // ssh's own last lines on stderr (login banners, refusals).
        std::string stderrTail(int lines = 8) const;
        // Every stderr line of ssh, on the reader thread. Set before open().
        std::function<void(const std::string&)> onStderrLine;

    private:
        bool readLine(std::string& line, std::chrono::steady_clock::time_point deadline, const std::function<bool()>& cancelled);
        CommandResult runLocked(const std::string& script, std::chrono::milliseconds timeout, const std::function<bool()>& cancelled);

        std::unique_ptr<ChildProcess> child_;
        std::mutex runMutex_;
        mutable std::mutex errMutex_;
        std::deque<std::string> errLines_;
        int socksPort_ = 0;
        std::string host_;
    };

    // A TCP port on 127.0.0.1 nobody listened on a moment ago.
    int freeLocalPort();

} // namespace sirius::app::ssh

#endif // SIRIUS_APP_REMOTE_HOST_HPP
