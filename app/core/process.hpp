#ifndef SIRIUS_APP_PROCESS_HPP
#define SIRIUS_APP_PROCESS_HPP

// A child process with its three standard streams on pipes: what the worker
// launcher needs to start `python -m sirius_worker`, read the one JSON line
// it prints once it listens, follow its stderr and stop it again, and what
// the Python environment's setup runs its installers with. CreateProcess on
// Windows, fork / exec elsewhere; no shell is involved on either, so an
// argument is passed as written. GUI-free: the GUI and sirius-cli share it.
//
// The child's stdin stays open for as long as the object lives (or until
// closeInput()): the worker runs with --exit-with-parent and stops when its
// stdin closes, so a crash of the application leaves no orphan holding the
// GPU.
//
// Threads: two reader threads of the object's own drain stdout and stderr.
// readLine() may be called from any one thread; the error handler is called
// on the stderr reader thread. start(), stop(), running(), waitForExit(),
// writeInput() and closeInput() are not safe against each other: the owner
// calls them from one thread at a time (the worker launcher holds a lock).
// start() copies the application's environment on the calling thread; a
// setenv() or putenv() on another thread at that moment is a data race, as
// it is for any reader of the environment. The application never changes
// its own environment for that reason: a child gets its variables through
// Options instead.

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

namespace sirius::app {

    class ChildProcess {
    public:
        struct Options {
            // A name without a directory is looked up on the PATH the child
            // gets (`environment` may replace it), with ".exe" added on
            // Windows unless the name ends in it. A relative path with a
            // directory is taken from `workingDirectory` when that is set,
            // on every platform.
            std::string program;                                          // resolved on PATH before start when it has no directory
            std::vector<std::string> arguments;
            std::string workingDirectory;                                 // empty = the parent's
            std::vector<std::pair<std::string, std::string>> environment; // added / replaced ("" = set empty)
            std::vector<std::string> unsetEnvironment;                    // removed from the inherited environment
            // A console Ctrl+C (SIGINT to the terminal's foreground group)
            // then reaches the application only, which decides what stops.
            bool ownProcessGroup = false;                                 // CREATE_NEW_PROCESS_GROUP | setpgid(0,0)
            // For installers, whose own children (ensurepip's pip, uv's
            // workers) must not outlive a cancel. stop() ends the whole tree
            // also when the child itself has already left; on Windows it
            // returns once every process of the tree has gone (2 s at most),
            // so that their files are closed. Where no job can hold the child
            // (a job this process runs in allows no nesting), it runs as
            // without killTree, and the first stderr line says so. Linux: the thread that calls start() must
            // outlive the child, since the signal follows that thread, not
            // the process.
            bool killTree = false;         // Windows: Job Object KILL_ON_JOB_CLOSE (CREATE_SUSPENDED, assign, resume);
                                           // POSIX: implies ownProcessGroup, stop() signals the group; Linux: PR_SET_PDEATHSIG
            // readLine() then ends only once both streams have ended.
            bool mergeErrorLines = false;  // stderr lines are also queued for readLine() (after the error handler)
        };

        ChildProcess();
        ~ChildProcess();   // stops the child
        ChildProcess(const ChildProcess&) = delete;
        ChildProcess& operator=(const ChildProcess&) = delete;

        // Called with every line the child writes to stderr. Set before start().
        // Before them it may get one line of this object's own, in
        // parentheses, about how the child runs (killTree without a job).
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
        bool programNotFound() const noexcept;     // the last start() failed because the program does not exist
        // Blocks while the pipe is full, that is while the child does not
        // read. A child that has closed its stdin makes it false (never
        // SIGPIPE).
        bool writeInput(const std::string& data);  // to the child's stdin; false once closed
        void closeInput();                         // end-of-file on the child's stdin
        // A negative timeout waits for as long as the child runs.
        bool waitForExit(int timeoutMs);           // true once the child ended; exitCode() is then valid

    private:
        struct Impl;
        void readLoop(bool errors);
        // A complete line from a reader: to the error handler, the queue or both.
        void deliver(bool errors, std::string line);
        // A reader's stream has ended.
        void streamEnded(bool errors);
        // `note`, unless empty, is delivered as the first stderr line.
        void startReaders(std::string note);
        void joinReaders();

        std::unique_ptr<Impl> impl_;
        std::function<void(const std::string&)> errorHandler_;
        std::thread outReader_, errReader_;
        std::mutex mutex_;
        std::condition_variable ready_;
        std::deque<std::string> lines_;     // complete stdout lines (and stderr lines with mergeErrorLines)
        int openQueuedStreams_ = 0;         // streams that still feed lines_, under mutex_
        bool outClosed_ = true;             // every stream that feeds lines_ has ended
        bool mergeErrorLines_ = false;      // set by start() before the readers run
        bool programNotFound_ = false;
        std::atomic<int> activeReaders_{0};
        std::atomic<bool> stopping_{false};
        int exitCode_ = -1;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_PROCESS_HPP
