#include "imgui/process.hpp"

#include <chrono>
#include <cstring>
#include <map>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace sirius::app::gui {

#ifdef _WIN32

    namespace {
        std::wstring widen(const std::string& s) {
            if (s.empty()) return std::wstring();
            const int n = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
            std::wstring w(static_cast<std::size_t>(n), L'\0');
            ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
            return w;
        }

        std::string narrow(const std::wstring& w) {
            if (w.empty()) return std::string();
            const int n = ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
            std::string s(static_cast<std::size_t>(n), '\0');
            ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
            return s;
        }

        // One argument as CommandLineToArgvW reads it back.
        std::wstring quoted(const std::wstring& arg) {
            if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) return arg;
            std::wstring out = L"\"";
            for (auto it = arg.begin();; ++it) {
                std::size_t backslashes = 0;
                while (it != arg.end() && *it == L'\\') {
                    ++it;
                    ++backslashes;
                }
                if (it == arg.end()) {
                    out.append(backslashes * 2, L'\\');
                    break;
                }
                if (*it == L'"') {
                    out.append(backslashes * 2 + 1, L'\\');
                    out.push_back(*it);
                } else {
                    out.append(backslashes, L'\\');
                    out.push_back(*it);
                }
            }
            out.push_back(L'"');
            return out;
        }

        std::string lastErrorText() {
            const DWORD code = ::GetLastError();
            wchar_t* buffer = nullptr;
            ::FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
                             0, reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
            std::string text = buffer ? narrow(buffer) : std::string("error ") + std::to_string(code);
            if (buffer) ::LocalFree(buffer);
            while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
            return text;
        }

        // The parent's environment with `extra` laid over it, as the block
        // CreateProcessW takes (sorted, case-insensitively, as it expects).
        std::wstring environmentBlock(const std::vector<std::pair<std::string, std::string>>& extra) {
            struct Less {
                bool operator()(const std::wstring& a, const std::wstring& b) const { return ::_wcsicmp(a.c_str(), b.c_str()) < 0; }
            };
            std::map<std::wstring, std::wstring, Less> vars;
            if (wchar_t* block = ::GetEnvironmentStringsW()) {
                for (const wchar_t* p = block; *p; p += std::wcslen(p) + 1) {
                    const std::wstring entry(p);
                    const std::size_t eq = entry.find(L'=', 1);   // "=C:=C:\dir" entries start with '='
                    if (eq != std::wstring::npos) vars[entry.substr(0, eq)] = entry.substr(eq + 1);
                }
                ::FreeEnvironmentStringsW(block);
            }
            for (const auto& kv : extra) vars[widen(kv.first)] = widen(kv.second);
            std::wstring out;
            for (const auto& kv : vars) {
                out += kv.first;
                out += L'=';
                out += kv.second;
                out.push_back(L'\0');
            }
            out.push_back(L'\0');
            return out;
        }
    } // namespace

    struct ChildProcess::Impl {
        HANDLE process = nullptr;
        HANDLE stdinWrite = nullptr;
        HANDLE stdoutRead = nullptr;
        HANDLE stderrRead = nullptr;
    };

    bool ChildProcess::start(const Options& options, std::string* error) {
        stop(0);
        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof sa;
        sa.bInheritHandle = TRUE;
        HANDLE inRead = nullptr, inWrite = nullptr, outRead = nullptr, outWrite = nullptr, errRead = nullptr, errWrite = nullptr;
        const auto closeAll = [&] {
            for (HANDLE h : {inRead, inWrite, outRead, outWrite, errRead, errWrite})
                if (h) ::CloseHandle(h);
        };
        if (!::CreatePipe(&inRead, &inWrite, &sa, 0) || !::CreatePipe(&outRead, &outWrite, &sa, 0) ||
            !::CreatePipe(&errRead, &errWrite, &sa, 0)) {
            if (error) *error = "cannot create pipes: " + lastErrorText();
            closeAll();
            return false;
        }
        // our ends are not for the child
        ::SetHandleInformation(inWrite, HANDLE_FLAG_INHERIT, 0);
        ::SetHandleInformation(outRead, HANDLE_FLAG_INHERIT, 0);
        ::SetHandleInformation(errRead, HANDLE_FLAG_INHERIT, 0);

        std::wstring command = quoted(widen(options.program));
        for (const std::string& a : options.arguments) command += L" " + quoted(widen(a));
        std::wstring env = environmentBlock(options.environment);
        const std::wstring cwd = widen(options.workingDirectory);

        // Only the child's three ends are handed to it. bInheritHandles alone
        // would pass every inheritable handle of the application as well:
        // files the CRT opened (a recording still being written) and sockets
        // (an RPC connection), which the worker would then keep open for the
        // rest of the session. The list must outlive the attribute list.
        HANDLE inherited[3] = {inRead, outWrite, errWrite};
        SIZE_T attributesSize = 0;
        ::InitializeProcThreadAttributeList(nullptr, 1, 0, &attributesSize);   // fails, and says how much it needs
        std::vector<unsigned char> attributesBuffer(attributesSize);
        const auto attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributesBuffer.data());
        if (!attributes || !::InitializeProcThreadAttributeList(attributes, 1, 0, &attributesSize)) {
            if (error) *error = lastErrorText();
            closeAll();
            return false;
        }
        if (!::UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited, sizeof inherited, nullptr,
                                         nullptr)) {
            if (error) *error = lastErrorText();
            ::DeleteProcThreadAttributeList(attributes);
            closeAll();
            return false;
        }

        STARTUPINFOEXW si{};
        si.StartupInfo.cb = sizeof si;
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        si.StartupInfo.hStdInput = inRead;
        si.StartupInfo.hStdOutput = outWrite;
        si.StartupInfo.hStdError = errWrite;
        si.lpAttributeList = attributes;
        PROCESS_INFORMATION pi{};
        const BOOL ok = ::CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                                         CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT, env.data(),
                                         cwd.empty() ? nullptr : cwd.c_str(), &si.StartupInfo, &pi);
        if (!ok) {
            if (error) *error = lastErrorText();
            ::DeleteProcThreadAttributeList(attributes);
            closeAll();
            return false;
        }
        ::DeleteProcThreadAttributeList(attributes);
        ::CloseHandle(pi.hThread);
        ::CloseHandle(inRead);
        ::CloseHandle(outWrite);
        ::CloseHandle(errWrite);
        impl_->process = pi.hProcess;
        impl_->stdinWrite = inWrite;
        impl_->stdoutRead = outRead;
        impl_->stderrRead = errRead;
        exitCode_ = -1;
        {
            const std::lock_guard<std::mutex> g(mutex_);
            lines_.clear();
            outClosed_ = false;
        }
        stopping_.store(false);
        activeReaders_.store(2);
        outReader_ = std::thread([this] {
            readLoop(false);
            --activeReaders_;
        });
        errReader_ = std::thread([this] {
            readLoop(true);
            --activeReaders_;
        });
        return true;
    }

    bool ChildProcess::running() {
        if (!impl_->process) return false;
        if (::WaitForSingleObject(impl_->process, 0) == WAIT_TIMEOUT) return true;
        DWORD code = 0;
        if (::GetExitCodeProcess(impl_->process, &code)) exitCode_ = static_cast<int>(code);
        return false;
    }

    void ChildProcess::stop(int graceMs) {
        if (impl_->process) {
            if (impl_->stdinWrite) {
                ::CloseHandle(impl_->stdinWrite);   // --exit-with-parent: the worker leaves by itself
                impl_->stdinWrite = nullptr;
            }
            if (::WaitForSingleObject(impl_->process, static_cast<DWORD>(graceMs > 0 ? graceMs : 0)) == WAIT_TIMEOUT) {
                ::TerminateProcess(impl_->process, 1);
                ::WaitForSingleObject(impl_->process, 1000);
            }
            DWORD code = 0;
            if (::GetExitCodeProcess(impl_->process, &code)) exitCode_ = static_cast<int>(code);
        }
        joinReaders();
        for (HANDLE* h : {&impl_->process, &impl_->stdinWrite, &impl_->stdoutRead, &impl_->stderrRead}) {
            if (*h) ::CloseHandle(*h);
            *h = nullptr;
        }
    }

    void ChildProcess::readLoop(bool errors) {
        const HANDLE h = errors ? impl_->stderrRead : impl_->stdoutRead;
        std::string pending;
        char buffer[4096];
        for (;;) {
            DWORD n = 0;
            if (!::ReadFile(h, buffer, sizeof buffer, &n, nullptr) || n == 0) break;
            pending.append(buffer, n);
            std::size_t nl;
            while ((nl = pending.find('\n')) != std::string::npos) {
                std::string line = pending.substr(0, nl);
                pending.erase(0, nl + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (errors) {
                    if (errorHandler_) errorHandler_(line);
                } else {
                    {
                        const std::lock_guard<std::mutex> g(mutex_);
                        lines_.push_back(std::move(line));
                    }
                    ready_.notify_all();
                }
            }
        }
        if (errors) {
            if (!pending.empty() && errorHandler_) errorHandler_(pending);
            return;
        }
        {
            const std::lock_guard<std::mutex> g(mutex_);
            if (!pending.empty()) lines_.push_back(std::move(pending));
            outClosed_ = true;
        }
        ready_.notify_all();
    }

#else   // POSIX

    namespace {
        // A pipe that is close-on-exec from the start: no other program the
        // application starts (a browser from openUrl) may inherit the
        // worker's stdin, or the worker would not see it close when the
        // application ends. Where pipe2 exists this is atomic, so a fork on
        // another thread cannot catch the ends without the flag either.
        bool closeOnExecPipe(int fds[2]) {
#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
            return ::pipe2(fds, O_CLOEXEC) == 0;
#else   // macOS has no pipe2
            if (::pipe(fds) != 0) return false;
            ::fcntl(fds[0], F_SETFD, FD_CLOEXEC);
            ::fcntl(fds[1], F_SETFD, FD_CLOEXEC);
            return true;
#endif
        }

        // In the child, between fork and exec (async-signal-safe calls only).
        // A pipe end that already is the stream it goes to (the application
        // was started with that stream closed) is not dup2'ed onto itself,
        // which would keep its close-on-exec flag: the flag is cleared instead.
        void redirect(int from, int to) {
            if (from == to) ::fcntl(to, F_SETFD, 0);
            else ::dup2(from, to);
        }

        // In the child, between fork and exec: every descriptor above the
        // standard streams closes at exec. Files and sockets the application
        // has open (a recording, an RPC connection) are not the child's, and a
        // socket it held would not close when the application closes it.
        // `keep` (close-on-exec already, so exec closes it too) must stay
        // usable until then; `limit` bounds the loop where close_range is
        // missing.
        void closeOthersOnExec(int keep, int limit) {
#ifdef CLOSE_RANGE_CLOEXEC
            if (::close_range(3, ~0U, CLOSE_RANGE_CLOEXEC) == 0) return;
#endif
            for (int fd = 3; fd < limit; ++fd)
                if (fd != keep) ::close(fd);
        }
    } // namespace

    struct ChildProcess::Impl {
        pid_t pid = -1;
        int stdinWrite = -1;
        int stdoutRead = -1;
        int stderrRead = -1;
    };

    bool ChildProcess::start(const Options& options, std::string* error) {
        stop(0);
        int in[2] = {-1, -1}, out[2] = {-1, -1}, err[2] = {-1, -1}, status[2] = {-1, -1};
        const auto closeAll = [&] {
            for (int fd : {in[0], in[1], out[0], out[1], err[0], err[1], status[0], status[1]})
                if (fd >= 0) ::close(fd);
        };
        // All four are close-on-exec: the child gets its three ends as its
        // standard streams (dup2 clears the flag), and exec closes the rest.
        // status[1] stays open until exec; a byte on it means exec failed.
        if (!closeOnExecPipe(in) || !closeOnExecPipe(out) || !closeOnExecPipe(err) || !closeOnExecPipe(status)) {
            if (error) *error = std::string("cannot create pipes: ") + std::strerror(errno);
            closeAll();
            return false;
        }

        std::vector<std::string> argStore;
        argStore.push_back(options.program);
        for (const std::string& a : options.arguments) argStore.push_back(a);
        std::vector<char*> argv;
        for (std::string& a : argStore) argv.push_back(a.data());
        argv.push_back(nullptr);
        // Read before fork: sysconf is not async-signal-safe. Descriptors are
        // handed out lowest first, so the application's own lie far below the
        // cap, which keeps the loop short where the limit is huge (containers).
        const long openMax = ::sysconf(_SC_OPEN_MAX);
        const int descriptorLimit = openMax > 0 && openMax < 65536 ? static_cast<int>(openMax) : 65536;

        const pid_t pid = ::fork();
        if (pid < 0) {
            if (error) *error = std::string("cannot fork: ") + std::strerror(errno);
            closeAll();
            return false;
        }
        if (pid == 0) {
            redirect(in[0], STDIN_FILENO);
            redirect(out[1], STDOUT_FILENO);
            redirect(err[1], STDERR_FILENO);
            closeOthersOnExec(status[1], descriptorLimit);
            // The application ignores SIGPIPE (main.cpp); an ignored signal
            // stays ignored across exec, and the worker should not start so.
            ::signal(SIGPIPE, SIG_DFL);
            for (const auto& kv : options.environment) ::setenv(kv.first.c_str(), kv.second.c_str(), 1);
            if (!options.workingDirectory.empty() && ::chdir(options.workingDirectory.c_str()) != 0) {
                const int e = errno;
                (void)!::write(status[1], &e, sizeof e);
                ::_exit(127);
            }
            ::execvp(argv[0], argv.data());
            const int e = errno;
            (void)!::write(status[1], &e, sizeof e);
            ::_exit(127);
        }
        ::close(in[0]);
        ::close(out[1]);
        ::close(err[1]);
        ::close(status[1]);
        int childErrno = 0;
        ssize_t got;
        do {
            got = ::read(status[0], &childErrno, sizeof childErrno);
        } while (got < 0 && errno == EINTR);
        ::close(status[0]);
        if (got > 0) {
            if (error) *error = std::strerror(childErrno);
            ::close(in[1]);
            ::close(out[0]);
            ::close(err[0]);
            int st = 0;
            ::waitpid(pid, &st, 0);
            return false;
        }
        impl_->pid = pid;
        impl_->stdinWrite = in[1];
        impl_->stdoutRead = out[0];
        impl_->stderrRead = err[0];
        exitCode_ = -1;
        {
            const std::lock_guard<std::mutex> g(mutex_);
            lines_.clear();
            outClosed_ = false;
        }
        stopping_.store(false);
        activeReaders_.store(2);
        outReader_ = std::thread([this] {
            readLoop(false);
            --activeReaders_;
        });
        errReader_ = std::thread([this] {
            readLoop(true);
            --activeReaders_;
        });
        return true;
    }

    bool ChildProcess::running() {
        if (impl_->pid < 0) return false;
        int st = 0;
        const pid_t r = ::waitpid(impl_->pid, &st, WNOHANG);
        if (r == 0) return true;
        if (r == impl_->pid) {
            exitCode_ = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
            impl_->pid = -1;
        }
        return false;
    }

    void ChildProcess::stop(int graceMs) {
        if (impl_->stdinWrite >= 0) {
            ::close(impl_->stdinWrite);
            impl_->stdinWrite = -1;
        }
        if (impl_->pid >= 0) {
            const auto waitFor = [this](int ms) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
                while (running()) {
                    if (std::chrono::steady_clock::now() >= deadline) return false;
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                return true;
            };
            if (!waitFor(graceMs / 2)) {
                ::kill(impl_->pid, SIGTERM);
                if (!waitFor(graceMs / 2 + 200)) {
                    ::kill(impl_->pid, SIGKILL);
                    waitFor(1000);
                }
            }
        }
        joinReaders();
        for (int* fd : {&impl_->stdoutRead, &impl_->stderrRead}) {
            if (*fd >= 0) ::close(*fd);
            *fd = -1;
        }
    }

    void ChildProcess::readLoop(bool errors) {
        const int fd = errors ? impl_->stderrRead : impl_->stdoutRead;
        std::string pending;
        char buffer[4096];
        for (;;) {
            struct pollfd pfd{};
            pfd.fd = fd;
            pfd.events = POLLIN;
            const int ready = ::poll(&pfd, 1, 200);
            if (ready < 0 && errno == EINTR) continue;
            if (ready == 0) {
                if (stopping_.load()) break;
                continue;
            }
            const ssize_t n = ::read(fd, buffer, sizeof buffer);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            pending.append(buffer, static_cast<std::size_t>(n));
            std::size_t nl;
            while ((nl = pending.find('\n')) != std::string::npos) {
                std::string line = pending.substr(0, nl);
                pending.erase(0, nl + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (errors) {
                    if (errorHandler_) errorHandler_(line);
                } else {
                    {
                        const std::lock_guard<std::mutex> g(mutex_);
                        lines_.push_back(std::move(line));
                    }
                    ready_.notify_all();
                }
            }
        }
        if (errors) {
            if (!pending.empty() && errorHandler_) errorHandler_(pending);
            return;
        }
        {
            const std::lock_guard<std::mutex> g(mutex_);
            if (!pending.empty()) lines_.push_back(std::move(pending));
            outClosed_ = true;
        }
        ready_.notify_all();
    }

#endif

    ChildProcess::ChildProcess() : impl_(std::make_unique<Impl>()) {}

    ChildProcess::~ChildProcess() { stop(); }

    void ChildProcess::setErrorHandler(std::function<void(const std::string& line)> handler) {
        errorHandler_ = std::move(handler);
    }

    bool ChildProcess::readLine(std::string& line, int timeoutMs) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this] { return !lines_.empty() || outClosed_; });
        if (lines_.empty()) return false;
        line = std::move(lines_.front());
        lines_.pop_front();
        return true;
    }

    void ChildProcess::joinReaders() {
        // The readers end when the child's ends of the pipes close, which its
        // exit does. A grandchild that inherited them (pip, started by the
        // worker) can keep them open, so after a moment for what is still in
        // the pipes the readers are told to stop rather than waited for.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (activeReaders_.load() > 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        stopping_.store(true);
#ifdef _WIN32
        if (activeReaders_.load() > 0) {
            if (impl_->stdoutRead) ::CancelIoEx(impl_->stdoutRead, nullptr);
            if (impl_->stderrRead) ::CancelIoEx(impl_->stderrRead, nullptr);
        }
#endif
        if (outReader_.joinable()) outReader_.join();
        if (errReader_.joinable()) errReader_.join();
    }

} // namespace sirius::app::gui
