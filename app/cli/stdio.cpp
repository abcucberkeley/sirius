#include "cli/stdio.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace sirius::cli {

    namespace {

        // The kept stdout. Until takeOverStdout() runs it is the ordinary one.
        int keptFd = 1;
        std::atomic<bool> outputBroken{false};

        std::mutex& outputMutex() {
            static std::mutex m;
            return m;
        }

        std::mutex& errorMutex() {
            static std::mutex m;
            return m;
        }

        struct Handlers {
            std::mutex mutex;
            std::function<void()> interrupt, terminate, cleanup;
        };

        Handlers& handlers() {
            static Handlers h;
            return h;
        }

        std::atomic<int> interrupts{0};

#ifdef _WIN32
        UINT savedOutputCodePage = 0;
#endif

        // Writes all of it, or gives up for good: a reader that went away does
        // not come back, and a server then stops (outputClosed()).
        void writeAll(int fd, const char* data, std::size_t size) {
            if (fd < 0 || outputBroken.load()) return;
            while (size > 0) {
#ifdef _WIN32
                const unsigned chunk = static_cast<unsigned>(size > (1u << 30) ? (1u << 30) : size);
                const int n = ::_write(fd, data, chunk);
#else
                const ssize_t n = ::write(fd, data, size);
                if (n < 0 && errno == EINTR) continue;
#endif
                if (n <= 0) {
                    outputBroken.store(true);
                    return;
                }
                data += n;
                size -= static_cast<std::size_t>(n);
            }
        }

        int readSome(char* buffer, unsigned size) {
#ifdef _WIN32
            return ::_read(0, buffer, size);
#else
            for (;;) {
                const ssize_t n = ::read(0, buffer, size);
                if (n < 0 && errno == EINTR) continue;
                return static_cast<int>(n);
            }
#endif
        }

        void runInterrupt() {
            // The first Ctrl+C cancels what runs and lets the command finish
            // in order; a second one means now.
            if (interrupts.fetch_add(1) == 0) {
                std::function<void()> f;
                {
                    const std::lock_guard<std::mutex> g(handlers().mutex);
                    f = handlers().interrupt;
                }
                if (f) f();
                return;
            }
            // exitProcess runs the emergency clean-up.
            writeError("sirius-cli: interrupted\n");
            exitProcess(130);
        }

        bool runTerminate() {
            std::function<void()> f;
            {
                const std::lock_guard<std::mutex> g(handlers().mutex);
                f = handlers().terminate;
            }
            if (!f) return false;
            f();
            return true;
        }

#ifdef _WIN32
        // A console, or the pipe a Cygwin / MSYS2 terminal (mintty, Git Bash)
        // gives a native program instead of one: a person is typing there.
        bool isTerminal(HANDLE h) {
            if (h == nullptr || h == INVALID_HANDLE_VALUE) return false;
            DWORD mode = 0;
            if (::GetConsoleMode(h, &mode)) return true;
            if (::GetFileType(h) != FILE_TYPE_PIPE) return false;
            std::vector<unsigned char> buffer(sizeof(FILE_NAME_INFO) + MAX_PATH * sizeof(WCHAR));
            if (!::GetFileInformationByHandleEx(h, FileNameInfo, buffer.data(), static_cast<DWORD>(buffer.size()))) return false;
            const auto* info = reinterpret_cast<const FILE_NAME_INFO*>(buffer.data());
            const std::wstring name(info->FileName, info->FileNameLength / sizeof(WCHAR));
            const bool cygwin = name.find(L"msys-") != std::wstring::npos || name.find(L"cygwin-") != std::wstring::npos;
            return cygwin && name.find(L"-pty") != std::wstring::npos;
        }

        BOOL WINAPI consoleHandler(DWORD type) {
            switch (type) {
                case CTRL_C_EVENT: runInterrupt(); return TRUE;
                case CTRL_BREAK_EVENT:
                case CTRL_CLOSE_EVENT:
                case CTRL_LOGOFF_EVENT:
                case CTRL_SHUTDOWN_EVENT:
                    if (!runTerminate()) return FALSE;
                    // The system ends the process about five seconds after a
                    // close event once this handler returns. The main thread
                    // cancels, stops the worker, removes the scratch directory
                    // and ends the process itself; this thread gives it the
                    // time to.
                    ::Sleep(4000);
                    return TRUE;
                default: return FALSE;
            }
        }

        void installConsoleHandler() {
            static std::once_flag once;
            std::call_once(once, [] { ::SetConsoleCtrlHandler(consoleHandler, TRUE); });
        }
#else
        // Signal handlers may only do async-signal-safe things: they write one
        // byte into a pipe, and a thread of ours reads it and acts.
        int signalPipe[2] = {-1, -1};

        void onSignal(int sig) {
            const int saved = errno;
            const char c = sig == SIGINT ? 'i' : 't';
            if (signalPipe[1] >= 0) {
                const ssize_t n = ::write(signalPipe[1], &c, 1);
                (void)n;
            }
            errno = saved;
        }

        void startSignalWatcher() {
            static std::once_flag once;
            std::call_once(once, [] {
                if (::pipe(signalPipe) != 0) return;
                for (int fd : signalPipe) ::fcntl(fd, F_SETFD, FD_CLOEXEC);
                ::fcntl(signalPipe[1], F_SETFL, ::fcntl(signalPipe[1], F_GETFL) | O_NONBLOCK);
                std::thread([] {
                    for (;;) {
                        char c = 0;
                        const ssize_t n = ::read(signalPipe[0], &c, 1);
                        if (n < 0 && errno == EINTR) continue;
                        if (n <= 0) return;
                        if (c == 'i') runInterrupt();
                        else runTerminate();
                    }
                }).detach();
            });
        }

        void catchSignal(int sig) {
            struct sigaction sa{};
            sa.sa_handler = onSignal;
            sigemptyset(&sa.sa_mask);
            // Reads of stdin and the like resume rather than fail with EINTR.
            sa.sa_flags = SA_RESTART;
            ::sigaction(sig, &sa, nullptr);
        }
#endif

    } // namespace

    void takeOverStdout() {
        std::fflush(stdout);
#ifdef _WIN32
        keptFd = ::_dup(1);
        if (keptFd >= 0) {
            // A line ends in "\n", never "\r\n".
            static_cast<void>(::_setmode(keptFd, _O_BINARY));
            // Child processes (the Python worker, installers) must not hold
            // our stdout open: a client waits for it to close.
            const HANDLE kept = reinterpret_cast<HANDLE>(::_get_osfhandle(keptFd));
            if (kept != INVALID_HANDLE_VALUE) ::SetHandleInformation(kept, HANDLE_FLAG_INHERIT, 0);
        }
        static_cast<void>(::_setmode(0, _O_BINARY));
        if (::_dup2(2, 1) == 0) {
            ::SetStdHandle(STD_OUTPUT_HANDLE, ::GetStdHandle(STD_ERROR_HANDLE));
        } else {
            // No stderr either: stray output goes nowhere rather than into the protocol.
            const int nul = ::_open("NUL", _O_WRONLY);
            if (nul >= 0) {
                static_cast<void>(::_dup2(nul, 1));
                ::_close(nul);
            }
        }
        // Paths and help pages are UTF-8; a console shows them right only
        // with this code page. The previous one is put back at exit.
        savedOutputCodePage = ::GetConsoleOutputCP();
        if (savedOutputCodePage != 0) ::SetConsoleOutputCP(CP_UTF8);
#else
        keptFd = ::fcntl(1, F_DUPFD_CLOEXEC, 3);
        if (::fcntl(2, F_GETFD) != -1) {
            ::dup2(2, 1);
        } else {
            const int nul = ::open("/dev/null", O_WRONLY);
            if (nul >= 0) {
                ::dup2(nul, 1);
                ::close(nul);
            }
        }
        // A client that closes the pipe must not kill us mid-cleanup: the
        // write fails with EPIPE instead, and outputClosed() says so.
        std::signal(SIGPIPE, SIG_IGN);
#endif
    }

    void writeLine(const std::string& line) {
        std::string buffer;
        buffer.reserve(line.size() + 1);
        buffer += line;
        buffer += '\n';
        const std::lock_guard<std::mutex> g(outputMutex());
        writeAll(keptFd, buffer.data(), buffer.size());
    }

    void writeRaw(const std::string& text) {
        const std::lock_guard<std::mutex> g(outputMutex());
        writeAll(keptFd, text.data(), text.size());
    }

    void writeError(const std::string& text) {
        const std::lock_guard<std::mutex> g(errorMutex());
        std::fwrite(text.data(), 1, text.size(), stderr);
        std::fflush(stderr);
    }

    bool outputClosed() { return outputBroken.load(); }

    bool stdinIsTerminal() {
#ifdef _WIN32
        return isTerminal(::GetStdHandle(STD_INPUT_HANDLE));
#else
        return ::isatty(0) == 1;
#endif
    }

    bool stdoutIsTerminal() {
#ifdef _WIN32
        return keptFd >= 0 && isTerminal(reinterpret_cast<HANDLE>(::_get_osfhandle(keptFd)));
#else
        return keptFd >= 0 && ::isatty(keptFd) == 1;
#endif
    }

    bool stderrIsTerminal() {
#ifdef _WIN32
        return isTerminal(::GetStdHandle(STD_ERROR_HANDLE));
#else
        return ::isatty(2) == 1;
#endif
    }

    void installInterruptHandler(std::function<void()> onFirst) {
        {
            const std::lock_guard<std::mutex> g(handlers().mutex);
            handlers().interrupt = std::move(onFirst);
        }
#ifdef _WIN32
        installConsoleHandler();
#else
        startSignalWatcher();
        catchSignal(SIGINT);
#endif
    }

    void installTerminationHandler(std::function<void()> onTerminate) {
        {
            const std::lock_guard<std::mutex> g(handlers().mutex);
            handlers().terminate = std::move(onTerminate);
        }
#ifdef _WIN32
        installConsoleHandler();
#else
        startSignalWatcher();
        catchSignal(SIGTERM);
        catchSignal(SIGHUP);
#endif
    }

    void setEmergencyCleanup(std::function<void()> cleanup) {
        const std::lock_guard<std::mutex> g(handlers().mutex);
        handlers().cleanup = std::move(cleanup);
    }

    void startStdinReader(std::function<void(std::string)> onLine, std::function<void()> onEof) {
        std::thread([onLine = std::move(onLine), onEof = std::move(onEof)] {
            std::string pending;
            std::vector<char> buffer(std::size_t{1} << 16);
            auto deliver = [&](std::string line) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                // A blank line is no message; skipping it here keeps both
                // protocols from answering stray newlines.
                if (!line.empty()) onLine(std::move(line));
            };
            for (;;) {
                const int n = readSome(buffer.data(), static_cast<unsigned>(buffer.size()));
                if (n <= 0) break;
                pending.append(buffer.data(), static_cast<std::size_t>(n));
                std::size_t start = 0;
                for (std::size_t nl = pending.find('\n'); nl != std::string::npos; nl = pending.find('\n', start)) {
                    deliver(pending.substr(start, nl - start));
                    start = nl + 1;
                }
                pending.erase(0, start);
            }
            if (!pending.empty()) deliver(std::move(pending));
            onEof();
        }).detach();
    }

    bool readStandardInput(std::string& out) {
        out.clear();
        std::vector<char> buffer(std::size_t{1} << 16);
        for (;;) {
            const int n = readSome(buffer.data(), static_cast<unsigned>(buffer.size()));
            if (n < 0) return false;
            if (n == 0) return true;
            out.append(buffer.data(), static_cast<std::size_t>(n));
        }
    }

    bool readAnswer(std::string& out) {
        out.clear();
        // Byte by byte, up to Enter: a terminal pipe may send a bare '\r'.
        for (;;) {
            char c = 0;
            const int n = readSome(&c, 1);
            if (n <= 0) return !out.empty();
            if (c == '\n' || c == '\r') return true;
            out += c;
        }
    }

    void exitProcess(int code) {
        // Whatever ends the process early (a second Ctrl+C, the timeout's
        // watchdog, an escaped exception) still removes the scratch
        // directory. Taken out under the lock, it runs once however many
        // threads get here; a normal end has cleared it already.
        std::function<void()> cleanup;
        {
            const std::lock_guard<std::mutex> g(handlers().mutex);
            cleanup = std::move(handlers().cleanup);
            handlers().cleanup = nullptr;
        }
        if (cleanup) {
            try {
                cleanup();
            } catch (...) {
                // Best effort: the exit code matters more.
            }
        }
        std::fflush(stdout);
        std::fflush(stderr);
#ifdef _WIN32
        if (savedOutputCodePage != 0) ::SetConsoleOutputCP(savedOutputCodePage);
#endif
        // No static destructors: a detached reader thread may still be
        // blocked on stdin, and a worker thread may still hold what they
        // would destroy.
        std::_Exit(code);
    }

} // namespace sirius::cli
