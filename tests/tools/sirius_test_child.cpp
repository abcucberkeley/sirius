// sirius_test_child: what the child-process, Python-environment and worker
// tests start, in place of a shell or `cmake -E` (whose `cat -` needs a newer
// CMake than the project's minimum). Its first argument picks what it does:
//
//   echo-stdin               copies stdin to stdout, line by line, until end of file
//   print [--stderr|--stdout] <line>...
//                            writes each line; the switches pick the stream for the lines after them
//   sleep <ms>               sleeps, then exits 0
//   exit <code>              exits with that code
//   env <NAME>               prints the variable's value, or "<unset>"
//   spawn-grandchild <ms> [<own ms>]
//                            starts itself with `sleep <ms>`, prints that process's id, and sleeps
//                            <own ms> (default <ms>) before it exits 0
//   ids                      prints its own process id and process group ("0" for the group on Windows)
//
// Every line is flushed at once: the tests read the pipe while it runs.
// Exit code 2 for arguments it does not understand, 3 when spawning fails.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

    void writeLine(std::FILE* stream, const std::string& line) {
        std::fputs(line.c_str(), stream);
        std::fputc('\n', stream);
        std::fflush(stream);
    }

    void sleepFor(const char* ms) { std::this_thread::sleep_for(std::chrono::milliseconds(std::atol(ms))); }

    // The variable as the operating system holds it, so that one set to ""
    // is told apart from one that is not set (on Windows the C runtime's
    // getenv is not asked: its copy is not guaranteed to keep empty ones).
    bool variable(const char* name, std::string& value) {
#ifdef _WIN32
        const DWORD n = ::GetEnvironmentVariableA(name, nullptr, 0);
        if (n == 0) {
            if (::GetLastError() == ERROR_ENVVAR_NOT_FOUND) return false;
            value.clear();
            return true;
        }
        std::string buffer(n, '\0');
        const DWORD got = ::GetEnvironmentVariableA(name, buffer.data(), n);
        buffer.resize(got);
        value = buffer;
        return true;
#else
        const char* v = std::getenv(name);
        if (!v) return false;
        value = v;
        return true;
#endif
    }

    // Starts this program again as `sleep <ms>`, detached from our streams so
    // that it holds none of the pipes the test reads; 0 when that fails.
    long spawnSleeper(const char* self, const char* ms) {
#ifdef _WIN32
        (void)self;
        char path[MAX_PATH * 4] = {};
        if (::GetModuleFileNameA(nullptr, path, static_cast<DWORD>(sizeof path)) == 0) return 0;
        std::string command = std::string("\"") + path + "\" sleep " + ms;
        STARTUPINFOA si{};
        si.cb = sizeof si;
        PROCESS_INFORMATION pi{};
        if (!::CreateProcessA(path, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return 0;
        ::CloseHandle(pi.hThread);
        ::CloseHandle(pi.hProcess);
        return static_cast<long>(pi.dwProcessId);
#else
#ifdef __linux__
        const char* program = "/proc/self/exe";
        (void)self;
#else
        const char* program = self;
#endif
        const pid_t pid = ::fork();
        if (pid < 0) return 0;
        if (pid == 0) {
            const int null = ::open("/dev/null", O_RDWR);
            if (null >= 0) {
                ::dup2(null, 0);
                ::dup2(null, 1);
                ::dup2(null, 2);
            }
            ::execl(program, program, "sleep", ms, static_cast<char*>(nullptr));
            ::_exit(127);
        }
        return static_cast<long>(pid);
#endif
    }

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const std::string mode = argv[1];
    if (mode == "echo-stdin") {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            writeLine(stdout, line);
        }
        return 0;
    }
    if (mode == "print") {
        std::FILE* stream = stdout;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--stderr") == 0) stream = stderr;
            else if (std::strcmp(argv[i], "--stdout") == 0) stream = stdout;
            else writeLine(stream, argv[i]);
        }
        return 0;
    }
    if (mode == "sleep" && argc == 3) {
        sleepFor(argv[2]);
        return 0;
    }
    if (mode == "exit" && argc == 3) return std::atoi(argv[2]);
    if (mode == "env" && argc == 3) {
        std::string value;
        writeLine(stdout, variable(argv[2], value) ? value : std::string("<unset>"));
        return 0;
    }
    if (mode == "spawn-grandchild" && (argc == 3 || argc == 4)) {
        const long pid = spawnSleeper(argv[0], argv[2]);
        if (pid == 0) return 3;
        writeLine(stdout, std::to_string(pid));
        sleepFor(argv[argc - 1]);
        return 0;
    }
    if (mode == "ids") {
#ifdef _WIN32
        writeLine(stdout, std::to_string(::GetCurrentProcessId()) + " 0");
#else
        writeLine(stdout, std::to_string(::getpid()) + " " + std::to_string(::getpgrp()));
#endif
        return 0;
    }
    return 2;
}
