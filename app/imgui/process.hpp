#ifndef SIRIUS_IMGUI_PROCESS_HPP
#define SIRIUS_IMGUI_PROCESS_HPP

// A child process with its three standard streams on pipes: what the worker
// launcher needs to start `python -m sirius_worker`, read the one JSON line
// it prints once it listens, follow its stderr and stop it again. CreateProcess
// on Windows, fork / exec elsewhere; no shell is involved on either, so an
// argument is passed as written.
//
// The child's stdin stays open for as long as the object lives: the worker
// runs with --exit-with-parent and stops when its stdin closes, so a crash
// of the application leaves no orphan holding the GPU.
//
// Threads: two reader threads of the object's own drain stdout and stderr.
// readLine() may be called from any one thread; the error handler is called
// on the stderr reader thread.

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::app::gui {

    class ChildProcess {
    public:
        struct Options {
            std::string program;                 // looked up on PATH when it has no directory
            std::vector<std::string> arguments;
            std::string workingDirectory;        // empty = the parent's
            // Added to (or replacing entries of) the parent's environment.
            std::vector<std::pair<std::string, std::string>> environment;
        };

        ChildProcess();
        ~ChildProcess();   // stops the child
        ChildProcess(const ChildProcess&) = delete;
        ChildProcess& operator=(const ChildProcess&) = delete;

        // Called with every line the child writes to stderr. Set before start().
        void setErrorHandler(std::function<void(const std::string& line)> handler);

        // False (with `error` filled) when the program could not be started.
        bool start(const Options& options, std::string* error = nullptr);
        bool running();
        // The next line of the child's stdout (without its line ending). False
        // when none arrived within `timeoutMs` or the stream ended.
        bool readLine(std::string& line, int timeoutMs);
        // Closes the child's stdin and waits `graceMs` for it to leave, then
        // terminates it. Safe to call when nothing runs.
        void stop(int graceMs = 3000);
        // Valid once the child has ended; -1 before.
        int exitCode() const noexcept { return exitCode_; }

    private:
        struct Impl;
        void readLoop(bool errors);
        void joinReaders();

        std::unique_ptr<Impl> impl_;
        std::function<void(const std::string&)> errorHandler_;
        std::thread outReader_, errReader_;
        std::mutex mutex_;
        std::condition_variable ready_;
        std::deque<std::string> lines_;     // complete stdout lines
        bool outClosed_ = true;
        std::atomic<int> activeReaders_{0};
        std::atomic<bool> stopping_{false};
        int exitCode_ = -1;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_PROCESS_HPP
