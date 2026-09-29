#include "core/process.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
extern char** environ;
#endif

namespace sirius::app {

    namespace {
        // Hands every complete line in `pending` to `take`, without its line
        // ending, and keeps what follows the last one.
        template <typename Take>
        void completeLines(std::string& pending, Take&& take) {
            std::size_t nl;
            while ((nl = pending.find('\n')) != std::string::npos) {
                std::string line = pending.substr(0, nl);
                pending.erase(0, nl + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                take(std::move(line));
            }
        }
    } // namespace

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

        std::string errorText(DWORD code) {
            wchar_t* buffer = nullptr;
            ::FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
                             0, reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
            std::string text = buffer ? narrow(buffer) : std::string("error ") + std::to_string(code);
            if (buffer) ::LocalFree(buffer);
            while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
            return text;
        }

        std::string lastErrorText() { return errorText(::GetLastError()); }

        // Variable names are case-insensitive on Windows, and CreateProcessW
        // expects the block sorted that way.
        struct NoCase {
            bool operator()(const std::wstring& a, const std::wstring& b) const { return ::_wcsicmp(a.c_str(), b.c_str()) < 0; }
        };
        using Variables = std::map<std::wstring, std::wstring, NoCase>;

        // The parent's environment with `unsetEnvironment` removed and
        // `environment` laid over it.
        Variables childEnvironment(const ChildProcess::Options& options) {
            Variables vars;
            if (wchar_t* block = ::GetEnvironmentStringsW()) {
                for (const wchar_t* p = block; *p; p += std::wcslen(p) + 1) {
                    const std::wstring entry(p);
                    const std::size_t eq = entry.find(L'=', 1);   // "=C:=C:\dir" entries start with '='
                    if (eq != std::wstring::npos) vars[entry.substr(0, eq)] = entry.substr(eq + 1);
                }
                ::FreeEnvironmentStringsW(block);
            }
            for (const std::string& name : options.unsetEnvironment) vars.erase(widen(name));
            for (const auto& kv : options.environment) vars[widen(kv.first)] = widen(kv.second);
            return vars;
        }

        // As the block CreateProcessW takes.
        std::wstring environmentBlock(const Variables& vars) {
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

        bool endsWithNoCase(const std::wstring& s, const std::wstring& suffix) {
            return s.size() >= suffix.size() && ::_wcsicmp(s.c_str() + (s.size() - suffix.size()), suffix.c_str()) == 0;
        }

        // `program` as found on `path` (the child's), "" when it is on none of
        // its directories. CreateProcessW would search the application's
        // directory and the working directory first, and the parent's PATH
        // rather than the child's. A relative path with a directory is taken
        // from `cwd`, the child's working directory, as execve takes it on
        // POSIX; CreateProcessW alone would take it from the parent's.
        std::wstring resolveProgram(const std::wstring& program, const std::wstring& path, const std::wstring& cwd) {
            if (program.empty()) return std::wstring();
            if (program.find_first_of(L"/\\:") != std::wstring::npos) {
                const bool relative =
                    program.front() != L'/' && program.front() != L'\\' && !(program.size() >= 2 && program[1] == L':');
                if (!relative || cwd.empty()) return program;
                return cwd.back() == L'/' || cwd.back() == L'\\' ? cwd + program : cwd + L'\\' + program;
            }
            const std::wstring file = endsWithNoCase(program, L".exe") ? program : program + L".exe";
            std::size_t start = 0;
            while (start <= path.size()) {
                std::size_t end = path.find(L';', start);
                if (end == std::wstring::npos) end = path.size();
                std::wstring dir = path.substr(start, end - start);
                start = end + 1;
                if (dir.size() >= 2 && dir.front() == L'"' && dir.back() == L'"') dir = dir.substr(1, dir.size() - 2);
                if (dir.empty()) continue;
                if (dir.back() != L'\\' && dir.back() != L'/') dir.push_back(L'\\');
                const std::wstring candidate = dir + file;
                const DWORD attributes = ::GetFileAttributesW(candidate.c_str());
                if (attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY)) return candidate;
            }
            return std::wstring();
        }

        // A job that ends every process in it when its last handle closes:
        // on stop(), and also when the application itself is killed.
        HANDLE killOnCloseJob() {
            const HANDLE job = ::CreateJobObjectW(nullptr, nullptr);
            if (!job) return nullptr;
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!::SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits)) {
                ::CloseHandle(job);
                return nullptr;
            }
            return job;
        }

        // Ends every process in `job` and waits, `timeoutMs` at most, until
        // each has gone. Terminating is asynchronous, and what a process has
        // open (pip's files in the folder a cancelled setup removes next)
        // stays open until then: its handle signals only once its handles are
        // closed and its executable and DLLs unmapped. The job's own count
        // of active processes is no help: it drops to 0 at once.
        void endJob(HANDLE job, int timeoutMs) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
            const auto left = [&deadline] {
                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
                return ms > 0 ? static_cast<DWORD>(ms) : DWORD{0};
            };
            // From here on no process in the job can start another (its
            // CreateProcess fails with ERROR_NOT_ENOUGH_QUOTA), so the list
            // below is all there is; those that run are left running.
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
            limits.BasicLimitInformation.ActiveProcessLimit = 1;
            ::SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits);
            // The processes are opened before they are ended: the job stops
            // listing a process once it has been told to end, and the handle
            // keeps its id from going to another one. Room for far more than
            // an installer starts; a longer list comes back cut short.
            constexpr std::size_t room = 256;
            std::vector<ULONG_PTR> buffer(sizeof(JOBOBJECT_BASIC_PROCESS_ID_LIST) / sizeof(ULONG_PTR) + room);
            const auto list = reinterpret_cast<JOBOBJECT_BASIC_PROCESS_ID_LIST*>(buffer.data());
            std::vector<HANDLE> processes;
            if (::QueryInformationJobObject(job, JobObjectBasicProcessIdList, list, static_cast<DWORD>(buffer.size() * sizeof(ULONG_PTR)),
                                            nullptr) ||
                ::GetLastError() == ERROR_MORE_DATA) {
                const DWORD listed = std::min<DWORD>(list->NumberOfProcessIdsInList, static_cast<DWORD>(room));
                for (DWORD i = 0; i < listed; ++i) {
                    const HANDLE h = ::OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                                                   static_cast<DWORD>(list->ProcessIdList[i]));
                    if (!h) continue;   // gone already
                    // The id may have been given to another process since.
                    BOOL inJob = FALSE;
                    if (::IsProcessInJob(h, job, &inJob) && inJob) processes.push_back(h);
                    else ::CloseHandle(h);
                }
            }
            ::TerminateJobObject(job, 1);
            for (HANDLE h : processes) {
                ::WaitForSingleObject(h, left());
                ::CloseHandle(h);
            }
        }
    } // namespace

    struct ChildProcess::Impl {
        HANDLE process = nullptr;
        HANDLE job = nullptr;   // killTree: the child and everything it starts
        HANDLE stdinWrite = nullptr;
        HANDLE stdoutRead = nullptr;
        HANDLE stderrRead = nullptr;
    };

    bool ChildProcess::start(const Options& options, std::string* error) {
        stop(0);
        programNotFound_ = false;
        const Variables vars = childEnvironment(options);
        const auto path = vars.find(L"PATH");
        const std::wstring program = resolveProgram(widen(options.program), path == vars.end() ? std::wstring() : path->second,
                                                    widen(options.workingDirectory));
        if (program.empty()) {
            programNotFound_ = true;
            if (error) *error = "not found on PATH";
            return false;
        }

        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof sa;
        sa.bInheritHandle = TRUE;
        HANDLE inRead = nullptr, inWrite = nullptr, outRead = nullptr, outWrite = nullptr, errRead = nullptr, errWrite = nullptr;
        HANDLE job = nullptr;
        const auto closeAll = [&] {
            for (HANDLE h : {inRead, inWrite, outRead, outWrite, errRead, errWrite, job})
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

        std::wstring command = quoted(program);
        for (const std::string& a : options.arguments) command += L" " + quoted(widen(a));
        std::wstring env = environmentBlock(vars);
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

        DWORD flags = CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT;
        if (options.ownProcessGroup) flags |= CREATE_NEW_PROCESS_GROUP;
        // The child is put into the job before it runs a single instruction,
        // so nothing it starts can escape it.
        if (options.killTree) job = killOnCloseJob();
        if (job) flags |= CREATE_SUSPENDED;

        STARTUPINFOEXW si{};
        si.StartupInfo.cb = sizeof si;
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        si.StartupInfo.hStdInput = inRead;
        si.StartupInfo.hStdOutput = outWrite;
        si.StartupInfo.hStdError = errWrite;
        si.lpAttributeList = attributes;
        PROCESS_INFORMATION pi{};
        const BOOL ok = ::CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, flags, env.data(),
                                         cwd.empty() ? nullptr : cwd.c_str(), &si.StartupInfo, &pi);
        if (!ok) {
            const DWORD code = ::GetLastError();
            programNotFound_ = code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND;
            if (error) *error = errorText(code);
            ::DeleteProcThreadAttributeList(attributes);
            closeAll();
            return false;
        }
        ::DeleteProcThreadAttributeList(attributes);
        if (job) {
            // Fails where this process runs in a job that allows no nesting
            // (before Windows 8, or a job with UI limits). The child then
            // runs as it would without killTree.
            if (!::AssignProcessToJobObject(job, pi.hProcess)) {
                ::CloseHandle(job);
                job = nullptr;
            }
            ::ResumeThread(pi.hThread);
        }
        ::CloseHandle(pi.hThread);
        ::CloseHandle(inRead);
        ::CloseHandle(outWrite);
        ::CloseHandle(errWrite);
        impl_->process = pi.hProcess;
        impl_->job = job;
        impl_->stdinWrite = inWrite;
        impl_->stdoutRead = outRead;
        impl_->stderrRead = errRead;
        exitCode_ = -1;
        mergeErrorLines_ = options.mergeErrorLines;
        // Without a job (none could be made, or the child could not be put
        // into it) it is said once, where the caller logs the child's own
        // complaints: first among its stderr lines, which with
        // mergeErrorLines also reach readLine(). Without the system's reason,
        // whose "Access is denied" would read as a locked file to whoever
        // classifies the lines.
        std::string note;
        if (options.killTree && !job)
            note = "(" + narrow(program) + " could not be put into a job object: stopping it ends that process only, not what it started)";
        startReaders(std::move(note));
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
            closeInput();   // --exit-with-parent: the worker leaves by itself
            if (::WaitForSingleObject(impl_->process, static_cast<DWORD>(graceMs > 0 ? graceMs : 0)) == WAIT_TIMEOUT && !impl_->job) {
                ::TerminateProcess(impl_->process, 1);
                ::WaitForSingleObject(impl_->process, 1000);
            }
        }
        // With killTree the job ends the child if it still runs, and what it
        // started and left running; all have gone when this returns, since a
        // cancelled setup removes the folder they ran from next.
        if (impl_->job) endJob(impl_->job, 2000);
        if (impl_->process) {
            DWORD code = 0;
            if (::GetExitCodeProcess(impl_->process, &code)) exitCode_ = static_cast<int>(code);
        }
        joinReaders();
        for (HANDLE* h : {&impl_->process, &impl_->job, &impl_->stdinWrite, &impl_->stdoutRead, &impl_->stderrRead}) {
            if (*h) ::CloseHandle(*h);
            *h = nullptr;
        }
    }

    bool ChildProcess::writeInput(const std::string& data) {
        if (!impl_->stdinWrite) return false;
        std::size_t done = 0;
        while (done < data.size()) {
            const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(data.size() - done, 1u << 20));
            DWORD written = 0;
            if (!::WriteFile(impl_->stdinWrite, data.data() + done, chunk, &written, nullptr)) {
                closeInput();   // the child has closed its end, or ended
                return false;
            }
            done += written;
        }
        return true;
    }

    void ChildProcess::closeInput() {
        if (!impl_->stdinWrite) return;
        ::CloseHandle(impl_->stdinWrite);
        impl_->stdinWrite = nullptr;
    }

    bool ChildProcess::waitForExit(int timeoutMs) {
        if (!impl_->process) return true;
        if (::WaitForSingleObject(impl_->process, timeoutMs < 0 ? INFINITE : static_cast<DWORD>(timeoutMs)) == WAIT_TIMEOUT) return false;
        running();   // records the exit code
        return true;
    }

    void ChildProcess::readLoop(bool errors) {
        const HANDLE h = errors ? impl_->stderrRead : impl_->stdoutRead;
        std::string pending;
        char buffer[4096];
        for (;;) {
            DWORD n = 0;
            if (!::ReadFile(h, buffer, sizeof buffer, &n, nullptr) || n == 0) break;
            pending.append(buffer, n);
            completeLines(pending, [&](std::string line) { deliver(errors, std::move(line)); });
        }
        if (!pending.empty()) deliver(errors, std::move(pending));
        streamEnded(errors);
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

        std::string variableName(const std::string& entry) { return entry.substr(0, entry.find('=')); }

        // The environment the child gets, as "NAME=value" entries: ours with
        // `unsetEnvironment` removed and `environment` laid over it.
        std::vector<std::string> childEnvironment(const ChildProcess::Options& options) {
            std::vector<std::string> env;
            for (char** e = environ; e && *e; ++e) env.emplace_back(*e);
            const auto drop = [&env](const std::string& name) {
                env.erase(std::remove_if(env.begin(), env.end(), [&](const std::string& entry) { return variableName(entry) == name; }),
                          env.end());
            };
            for (const std::string& name : options.unsetEnvironment) drop(name);
            for (const auto& kv : options.environment) {
                drop(kv.first);
                env.push_back(kv.first + "=" + kv.second);
            }
            return env;
        }

        // `program` as execvp would find it on the child's PATH, "" when it
        // is on none of its directories. A name with a slash is left to
        // execve (relative to the working directory the child gets).
        std::string resolveProgram(const std::string& program, const std::vector<std::string>& env) {
            if (program.empty()) return std::string();
            if (program.find('/') != std::string::npos) return program;
            std::string path = "/bin:/usr/bin";   // what execvp searches when PATH is not set
            for (const std::string& entry : env)
                if (variableName(entry) == "PATH") path = entry.substr(5);
            std::size_t start = 0;
            while (start <= path.size()) {
                std::size_t end = path.find(':', start);
                if (end == std::string::npos) end = path.size();
                const std::string dir = path.substr(start, end - start);
                start = end + 1;
                // An empty entry would mean the working directory, which is
                // no place to pick an interpreter from.
                if (dir.empty()) continue;
                const std::string candidate = dir + "/" + program;
                struct stat st{};
                if (::stat(candidate.c_str(), &st) == 0 && S_ISREG(st.st_mode) && ::access(candidate.c_str(), X_OK) == 0) return candidate;
            }
            return std::string();
        }
    } // namespace

    struct ChildProcess::Impl {
        pid_t pid = -1;         // the child, until it is reaped
        pid_t group = -1;       // killTree: the child's process group, which stop() ends
        bool exited = false;    // killTree: seen to have ended, left unreaped until stop() (running())
        int stdinWrite = -1;
        int stdoutRead = -1;
        int stderrRead = -1;
    };

    bool ChildProcess::start(const Options& options, std::string* error) {
        stop(0);
        programNotFound_ = false;
        // Everything the child needs is made here, before fork. The child of
        // a process with other threads may only make async-signal-safe
        // calls until it execs: setenv, malloc and execvp's PATH search take
        // locks another thread may have held at the moment of the fork, and
        // the child would wait for them forever.
        std::vector<std::string> envStore = childEnvironment(options);
        const std::string program = resolveProgram(options.program, envStore);
        if (program.empty()) {
            programNotFound_ = true;
            if (error) *error = "not found on PATH";
            return false;
        }
        std::vector<char*> envp;
        for (std::string& e : envStore) envp.push_back(e.data());
        envp.push_back(nullptr);
        std::vector<std::string> argStore;
        argStore.push_back(options.program);
        for (const std::string& a : options.arguments) argStore.push_back(a);
        std::vector<char*> argv;
        for (std::string& a : argStore) argv.push_back(a.data());
        argv.push_back(nullptr);
        const char* const cwd = options.workingDirectory.empty() ? nullptr : options.workingDirectory.c_str();
        const bool ownGroup = options.ownProcessGroup || options.killTree;
#ifdef __linux__
        const bool deathSignal = options.killTree;
        const pid_t parent = ::getpid();
#endif
        // The child starts with no signal blocked, whatever the thread that
        // starts it blocks (a host that waits for signals on a thread of its
        // own blocks them everywhere else): SIGTERM from stop() must work.
        sigset_t noSignals;
        sigemptyset(&noSignals);
        // Read before fork: sysconf is not async-signal-safe. Descriptors are
        // handed out lowest first, so the application's own lie far below the
        // cap, which keeps the loop short where the limit is huge (containers).
        const long openMax = ::sysconf(_SC_OPEN_MAX);
        const int descriptorLimit = openMax > 0 && openMax < 65536 ? static_cast<int>(openMax) : 65536;

        int in[2] = {-1, -1}, out[2] = {-1, -1}, err[2] = {-1, -1}, status[2] = {-1, -1};
        const auto closeAll = [&] {
            for (int fd : {in[0], in[1], out[0], out[1], err[0], err[1], status[0], status[1]})
                if (fd >= 0) ::close(fd);
        };
        // All four are close-on-exec: the child gets its three ends as its
        // standard streams (dup2 clears the flag), and exec closes the rest.
        // status[1] stays open until exec; what arrives on it means that
        // chdir (1) or exec (2) failed, and with which errno.
        if (!closeOnExecPipe(in) || !closeOnExecPipe(out) || !closeOnExecPipe(err) || !closeOnExecPipe(status)) {
            if (error) *error = std::string("cannot create pipes: ") + std::strerror(errno);
            closeAll();
            return false;
        }
#ifdef F_SETNOSIGPIPE
        ::fcntl(in[1], F_SETNOSIGPIPE, 1);   // writeInput() to a child that closed its stdin
#endif

        const pid_t pid = ::fork();
        if (pid < 0) {
            if (error) *error = std::string("cannot fork: ") + std::strerror(errno);
            closeAll();
            return false;
        }
        if (pid == 0) {
            if (ownGroup) ::setpgid(0, 0);
#ifdef __linux__
            // Ends the child with the thread that started it, and so with the
            // application however it ends. The parent may already be gone.
            if (deathSignal) {
                ::prctl(PR_SET_PDEATHSIG, SIGKILL);
                if (::getppid() != parent) ::_exit(127);
            }
#endif
            redirect(in[0], STDIN_FILENO);
            redirect(out[1], STDOUT_FILENO);
            redirect(err[1], STDERR_FILENO);
            closeOthersOnExec(status[1], descriptorLimit);
            // The application ignores SIGPIPE (main.cpp); an ignored signal
            // stays ignored across exec, and the worker should not start so.
            ::signal(SIGPIPE, SIG_DFL);
            ::sigprocmask(SIG_SETMASK, &noSignals, nullptr);
            int report[2] = {0, 0};
            if (cwd && ::chdir(cwd) != 0) {
                report[0] = 1;
            } else {
                ::execve(program.c_str(), argv.data(), envp.data());
                report[0] = 2;
            }
            report[1] = errno;
            (void)!::write(status[1], report, sizeof report);
            ::_exit(127);
        }
        // Also here, so that a stop() right after start() already reaches the
        // group; it fails harmlessly once the child has exec'ed.
        if (ownGroup) ::setpgid(pid, pid);
        ::close(in[0]);
        ::close(out[1]);
        ::close(err[1]);
        ::close(status[1]);
        int report[2] = {0, 0};
        std::size_t got = 0;
        while (got < sizeof report) {
            const ssize_t n = ::read(status[0], reinterpret_cast<char*>(report) + got, sizeof report - got);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            got += static_cast<std::size_t>(n);
        }
        ::close(status[0]);
        if (got > 0) {
            programNotFound_ = report[0] == 2 && (report[1] == ENOENT || report[1] == ENOTDIR);
            if (error) {
                *error = std::strerror(report[1]);
                if (report[0] == 1) *error = "cannot change to " + options.workingDirectory + ": " + *error;
            }
            ::close(in[1]);
            ::close(out[0]);
            ::close(err[0]);
            int st = 0;
            while (::waitpid(pid, &st, 0) < 0 && errno == EINTR) {
            }
            return false;
        }
        impl_->pid = pid;
        impl_->group = options.killTree ? pid : -1;
        impl_->exited = false;
        impl_->stdinWrite = in[1];
        impl_->stdoutRead = out[0];
        impl_->stderrRead = err[0];
        exitCode_ = -1;
        mergeErrorLines_ = options.mergeErrorLines;
        startReaders(std::string());
        return true;
    }

    bool ChildProcess::running() {
        if (impl_->pid < 0 || impl_->exited) return false;
        if (impl_->group > 0) {
            // Seen, not reaped: the zombie keeps the child's pid, and with it
            // the process group's id, from being given to another process
            // before stop() has signalled that group.
            siginfo_t info{};
            if (::waitid(P_PID, static_cast<id_t>(impl_->pid), &info, WEXITED | WNOHANG | WNOWAIT) != 0) {
                if (errno == EINTR) return true;
                // Reaped by someone else (SIGCHLD set to SIG_IGN, a waitpid(-1)
                // elsewhere): gone, its exit code lost. Counted as seen, so
                // that stop() still signals the group its children are in.
                impl_->exited = true;
                return false;
            }
            if (info.si_pid == 0) return true;
            exitCode_ = info.si_code == CLD_EXITED ? info.si_status : 128 + info.si_status;
            impl_->exited = true;
            return false;
        }
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
        closeInput();
        if (impl_->pid >= 0) {
            const auto waitFor = [this](int ms) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
                while (running()) {
                    if (std::chrono::steady_clock::now() >= deadline) return false;
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                return true;
            };
            const pid_t target = impl_->group > 0 ? -impl_->group : impl_->pid;
            if (!waitFor(graceMs / 2)) {
                ::kill(target, SIGTERM);
                if (!waitFor(graceMs / 2 + 200)) {
                    ::kill(target, SIGKILL);
                    waitFor(1000);
                }
            }
            if (impl_->group > 0 && impl_->exited) {
                // What the child started and left running goes with it; then
                // the child is reaped, which frees its pid.
                ::kill(-impl_->group, SIGKILL);
                int st = 0;
                while (::waitpid(impl_->pid, &st, 0) < 0 && errno == EINTR) {
                }
                impl_->pid = -1;
            }
        }
        if (impl_->pid < 0) {
            impl_->group = -1;
            impl_->exited = false;
        }
        joinReaders();
        for (int* fd : {&impl_->stdoutRead, &impl_->stderrRead}) {
            if (*fd >= 0) ::close(*fd);
            *fd = -1;
        }
    }

    bool ChildProcess::writeInput(const std::string& data) {
        if (impl_->stdinWrite < 0) return false;
#ifndef F_SETNOSIGPIPE
        // A write to a pipe nobody reads raises SIGPIPE, which ends a process
        // that does not ignore it (a test binary, a host that forgot to). It
        // is blocked on this thread for the write, and one the write raised
        // is taken back before it is unblocked.
        sigset_t pipeSignal, previous, pending;
        sigemptyset(&pipeSignal);
        sigaddset(&pipeSignal, SIGPIPE);
        ::pthread_sigmask(SIG_BLOCK, &pipeSignal, &previous);
        sigemptyset(&pending);
        ::sigpending(&pending);
        const bool pendingBefore = sigismember(&pending, SIGPIPE) == 1;
#endif
        int failure = 0;
        std::size_t done = 0;
        while (done < data.size()) {
            const ssize_t n = ::write(impl_->stdinWrite, data.data() + done, data.size() - done);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) {
                failure = errno;
                break;
            }
            done += static_cast<std::size_t>(n);
        }
#ifndef F_SETNOSIGPIPE
        if (failure == EPIPE && !pendingBefore) {
            const struct timespec zero{0, 0};
            while (::sigtimedwait(&pipeSignal, nullptr, &zero) < 0 && errno == EINTR) {
            }
        }
        ::pthread_sigmask(SIG_SETMASK, &previous, nullptr);
#endif
        if (failure == 0) return true;
        closeInput();   // the child has closed its end, or ended
        return false;
    }

    void ChildProcess::closeInput() {
        if (impl_->stdinWrite < 0) return;
        ::close(impl_->stdinWrite);
        impl_->stdinWrite = -1;
    }

    bool ChildProcess::waitForExit(int timeoutMs) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs > 0 ? timeoutMs : 0);
        while (running()) {
            if (timeoutMs >= 0 && std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return true;
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
            completeLines(pending, [&](std::string line) { deliver(errors, std::move(line)); });
        }
        if (!pending.empty()) deliver(errors, std::move(pending));
        streamEnded(errors);
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

    bool ChildProcess::programNotFound() const noexcept { return programNotFound_; }

    void ChildProcess::deliver(bool errors, std::string line) {
        if (errors) {
            if (errorHandler_) errorHandler_(line);
            if (!mergeErrorLines_) return;
        }
        {
            const std::lock_guard<std::mutex> g(mutex_);
            lines_.push_back(std::move(line));
        }
        ready_.notify_all();
    }

    void ChildProcess::streamEnded(bool errors) {
        if (errors && !mergeErrorLines_) return;
        {
            const std::lock_guard<std::mutex> g(mutex_);
            if (--openQueuedStreams_ <= 0) outClosed_ = true;
        }
        ready_.notify_all();
    }

    void ChildProcess::startReaders(std::string note) {
        {
            const std::lock_guard<std::mutex> g(mutex_);
            lines_.clear();
            openQueuedStreams_ = mergeErrorLines_ ? 2 : 1;
            outClosed_ = false;
        }
        stopping_.store(false);
        activeReaders_.store(2);
        outReader_ = std::thread([this] {
            readLoop(false);
            --activeReaders_;
        });
        // The note goes out as the child's first stderr line, so that the
        // error handler is only ever called on this thread.
        errReader_ = std::thread([this, note = std::move(note)]() mutable {
            if (!note.empty()) deliver(true, std::move(note));
            readLoop(true);
            --activeReaders_;
        });
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

} // namespace sirius::app
