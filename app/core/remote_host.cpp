#include "core/remote_host.hpp"

#include "core/host.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
using sock_t = SOCKET;
#define SIRIUS_BAD_SOCKET INVALID_SOCKET
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
using sock_t = int;
#define SIRIUS_BAD_SOCKET (-1)
#endif

namespace sirius::app::ssh {

    using json = nlohmann::json;

    namespace {
        constexpr const char* kPortVar = "SIRIUS_ASKPASS_PORT";
        constexpr const char* kSecretVar = "SIRIUS_ASKPASS_SECRET";

#ifdef _WIN32
        struct WinsockInit {
            WinsockInit() {
                WSADATA d;
                if (WSAStartup(MAKEWORD(2, 2), &d) != 0) d = WSADATA{};
            }
            ~WinsockInit() { WSACleanup(); }
        };
        void ensureSockets() { static WinsockInit init; }
        void closeSock(sock_t s) { closesocket(s); }
#else
        void ensureSockets() {}
        void closeSock(sock_t s) { ::close(s); }
#endif

        // readable within `ms`
        bool readable(sock_t s, int ms) {
#ifdef _WIN32
            fd_set set;
            FD_ZERO(&set);
            FD_SET(s, &set);
            timeval tv{ms / 1000, (ms % 1000) * 1000};
            return select(0, &set, nullptr, nullptr, &tv) > 0;
#else
            pollfd p{s, POLLIN, 0};
            return ::poll(&p, 1, ms) > 0;
#endif
        }

        sock_t listenLoopback(int& port) {
            ensureSockets();
            sock_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (s == SIRIUS_BAD_SOCKET) return s;
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            a.sin_port = 0;
            if (::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0 || ::listen(s, 4) != 0) {
                closeSock(s);
                return SIRIUS_BAD_SOCKET;
            }
            socklen_t len = sizeof a;
            getsockname(s, reinterpret_cast<sockaddr*>(&a), &len);
            port = ntohs(a.sin_port);
            return s;
        }

        // One '\n'-terminated line (without it), at most `cap` bytes, within `ms`.
        bool readSockLine(sock_t s, std::string& line, int ms, std::size_t cap = 64 * 1024) {
            line.clear();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
            char c = 0;
            while (line.size() < cap) {
                const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
                if (left <= 0 || !readable(s, static_cast<int>(left))) return false;
                const auto n = ::recv(s, &c, 1, 0);
                if (n <= 0) return false;
                if (c == '\n') return true;
                line.push_back(c);
            }
            return false;
        }

        bool sendAll(sock_t s, const std::string& data) {
            std::size_t sent = 0;
            while (sent < data.size()) {
                const auto n = ::send(s, data.data() + sent, static_cast<int>(data.size() - sent), 0);
                if (n <= 0) return false;
                sent += static_cast<std::size_t>(n);
            }
            return true;
        }

        bool sameSecret(const std::string& a, const std::string& b) {
            if (a.size() != b.size()) return false;
            unsigned char diff = 0;
            for (std::size_t i = 0; i < a.size(); ++i) diff |= static_cast<unsigned char>(a[i] ^ b[i]);
            return diff == 0;
        }

        bool endsWith(const std::string& s, const std::string& tail) {
            return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
        }
    } // namespace

    // --- small helpers ----------------------------------------------------------------

    std::string findSsh() {
#ifdef _WIN32
        const std::string root = host::environment("SystemRoot");
        const std::string system = (root.empty() ? std::string("C:/Windows") : root) + "/System32/OpenSSH/ssh.exe";
        if (host::isFile(system)) return system;
#endif
        return host::findExecutable("ssh");
    }

    std::string shellQuote(const std::string& s) {
        std::string out = "'";
        for (char c : s) {
            if (c == '\'') out += "'\\''";
            else out.push_back(c);
        }
        return out + "'";
    }

    std::string remotePathWord(const std::string& path) {
        if (path == "~") return "\"$HOME\"";
        if (path.rfind("~/", 0) == 0) return "\"$HOME\"/" + shellQuote(path.substr(2));
        return shellQuote(path);
    }

    std::string randomHex(int bytes) {
        std::random_device rd;
        static const char* digits = "0123456789abcdef";
        std::string out;
        for (int i = 0; i < bytes; ++i) {
            const unsigned v = rd() & 0xffu;
            out.push_back(digits[v >> 4]);
            out.push_back(digits[v & 15]);
        }
        return out;
    }

    int freeLocalPort() {
        int port = 0;
        sock_t s = listenLoopback(port);
        if (s == SIRIUS_BAD_SOCKET) return 0;
        closeSock(s);
        return port;
    }

    // --- askpass --------------------------------------------------------------------

    AskpassServer::AskpassServer(Handler handler) : handler_(std::move(handler)), secret_(randomHex(24)) {
        const sock_t s = listenLoopback(port_);
        if (s == SIRIUS_BAD_SOCKET) throw SshError("cannot open a loopback port for the password prompts");
        listener_ = static_cast<std::intptr_t>(s);
        thread_ = std::thread([this] { serve(); });
    }

    AskpassServer::~AskpassServer() {
        stop_.store(true);
        if (thread_.joinable()) thread_.join();
        if (listener_ != -1) closeSock(static_cast<sock_t>(listener_));
    }

    std::vector<std::pair<std::string, std::string>> AskpassServer::environment(const std::string& program) const {
        return {{"SSH_ASKPASS", program},
                {"SSH_ASKPASS_REQUIRE", "force"},
                // OpenSSH before 8.4 wants a DISPLAY before it uses SSH_ASKPASS at all
                {"DISPLAY", host::hasEnvironment("DISPLAY") ? host::environment("DISPLAY") : std::string(":0")},
                {kPortVar, std::to_string(port_)},
                {kSecretVar, secret_}};
    }

    void AskpassServer::serve() {
        const sock_t listener = static_cast<sock_t>(listener_);
        while (!stop_.load()) {
            if (!readable(listener, 200)) continue;
            const sock_t c = ::accept(listener, nullptr, nullptr);
            if (c == SIRIUS_BAD_SOCKET) continue;
            std::string line;
            json reply = {{"cancel", true}};
            if (readSockLine(c, line, 5000)) {
                try {
                    const json req = json::parse(line);
                    if (sameSecret(req.value("secret", std::string()), secret_)) {
                        Prompt p;
                        p.text = req.value("prompt", std::string());
                        const std::string kind = req.value("kind", std::string());
                        p.notifyOnly = kind == "none";
                        p.echo = kind == "confirm" || p.text.find("(yes/no") != std::string::npos;
                        const std::optional<std::string> answer = handler_ ? handler_(p) : std::nullopt;
                        if (answer) reply = {{"answer", *answer}};
                    }
                } catch (const std::exception&) {
                    // a malformed request is answered as cancelled
                }
            }
            sendAll(c, reply.dump() + "\n");
            closeSock(c);
        }
    }

    bool isAskpassInvocation() { return host::hasEnvironment(kPortVar) && host::hasEnvironment(kSecretVar); }

    int askpassMain(int argc, char** argv) {
        const std::string prompt = argc > 1 && argv[1] ? argv[1] : std::string();
        const int port = std::atoi(host::environment(kPortVar).c_str());
        if (port <= 0) return 1;
        ensureSockets();
        sock_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s == SIRIUS_BAD_SOCKET) return 1;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = htons(static_cast<unsigned short>(port));
        if (::connect(s, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) {
            closeSock(s);
            return 1;
        }
        const json req = {{"secret", host::environment(kSecretVar)}, {"prompt", prompt}, {"kind", host::environment("SSH_ASKPASS_PROMPT")}};
        std::string line;
        // the user may take a while: a one-time code from a phone
        const bool got = sendAll(s, req.dump() + "\n") && readSockLine(s, line, 15 * 60 * 1000, 1 << 20);
        closeSock(s);
        if (!got) return 1;
        std::string answer;
        try {
            const json r = json::parse(line);
            if (!r.contains("answer") || !r["answer"].is_string()) return 1;
            answer = r["answer"].get<std::string>() + "\n";
        } catch (const std::exception&) {
            return 1;
        }
#ifdef _WIN32
        // a GUI-subsystem executable has no CRT stdout to speak of: the handle ssh gave it
        DWORD written = 0;
        WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), answer.data(), static_cast<DWORD>(answer.size()), &written, nullptr);
#else
        std::fwrite(answer.data(), 1, answer.size(), stdout);
        std::fflush(stdout);
#endif
        std::fill(answer.begin(), answer.end(), '\0');
        return 0;
    }

    // --- the session ----------------------------------------------------------------

    std::vector<std::string> sshArguments(const Options& o, int socksPort) {
        std::vector<std::string> a = {"-T",
                                      "-o", "BatchMode=no",
                                      // a wrong password costs one attempt, never three
                                      "-o", "NumberOfPasswordPrompts=1",
                                      "-o", "ConnectTimeout=15",
                                      "-o", "ExitOnForwardFailure=yes",
                                      "-o", "ServerAliveInterval=30",
                                      "-o", "ServerAliveCountMax=3"};
        if (socksPort > 0) {
            a.push_back("-D");
            a.push_back("127.0.0.1:" + std::to_string(socksPort));
        }
        for (const std::string& e : o.extraOptions) {
            a.push_back("-o");
            a.push_back(e);
        }
        a.push_back("--");
        a.push_back(o.host);
        a.push_back(o.remoteCommand);
        return a;
    }

    Session::Session() = default;
    Session::~Session() { close(); }

    void Session::open(const Options& o, const std::function<bool()>& cancelled, std::chrono::milliseconds timeout) {
        close();
        if (o.host.empty()) throw SshError("no SSH host given");
        if (o.host.front() == '-') throw SshError("'" + o.host + "' is not an SSH host");
        const std::string program = o.program.empty() ? findSsh() : o.program;
        if (program.empty()) throw SshError("no ssh client found: install OpenSSH (on Windows: Settings > Optional features > OpenSSH Client)");
        socksPort_ = o.socksPort < 0 ? freeLocalPort() : o.socksPort;
        host_ = o.host;
        {
            const std::lock_guard<std::mutex> g(errMutex_);
            errLines_.clear();
        }
        ChildProcess::Options co;
        co.program = program;
        co.arguments = o.programArgs;
        const std::vector<std::string> args = sshArguments(o, socksPort_);
        co.arguments.insert(co.arguments.end(), args.begin(), args.end());
        co.environment = o.environment;
        child_ = std::make_unique<ChildProcess>();
        child_->setErrorHandler([this](const std::string& line) {
            std::string l = line;
            if (!l.empty() && l.back() == '\r') l.pop_back();
            {
                const std::lock_guard<std::mutex> g(errMutex_);
                errLines_.push_back(l);
                while (errLines_.size() > 200) errLines_.pop_front();
            }
            if (onStderrLine) onStderrLine(l);
        });
        std::string error;
        if (!child_->start(co, &error)) {
            child_.reset();
            throw SshError("cannot start " + program + ": " + error);
        }
        // The first command is answered once the login is over, prompts and all.
        try {
            const std::lock_guard<std::mutex> g(runMutex_);
            const CommandResult r = runLocked("printf 'sirius-ready\\n'", timeout, cancelled);
            if (r.out.find("sirius-ready") == std::string::npos) throw SshError("the remote shell did not answer", r.err);
        } catch (const SshError& e) {
            const std::string tail = stderrTail();
            close();
            throw SshError(e.what(), tail.empty() ? e.detail : tail);
        }
    }

    bool Session::isOpen() { return child_ && child_->running(); }

    void Session::close() {
        if (child_) {
            child_->stop(1000);
            child_.reset();
        }
    }

    std::string Session::stderrTail(int lines) const {
        const std::lock_guard<std::mutex> g(errMutex_);
        std::string out;
        const int n = static_cast<int>(errLines_.size());
        for (int i = std::max(0, n - lines); i < n; ++i) {
            if (errLines_[static_cast<std::size_t>(i)].empty()) continue;
            if (!out.empty()) out += "\n";
            out += errLines_[static_cast<std::size_t>(i)];
        }
        return out;
    }

    bool Session::readLine(std::string& line, std::chrono::steady_clock::time_point deadline, const std::function<bool()>& cancelled) {
        for (;;) {
            if (!child_) throw SshError("the SSH connection is closed");
            if (child_->readLine(line, 200)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                return true;
            }
            if (!child_->running()) {
                // what is still queued is read before the end is reported
                if (child_->readLine(line, 0)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    return true;
                }
                throw SshError("the SSH connection ended");
            }
            if (cancelled && cancelled()) throw SshError("cancelled");
            if (std::chrono::steady_clock::now() > deadline) return false;
        }
    }

    CommandResult Session::run(const std::string& script, std::chrono::milliseconds timeout, const std::function<bool()>& cancelled) {
        const std::lock_guard<std::mutex> g(runMutex_);
        if (!isOpen()) throw SshError("the SSH connection is closed");
        return runLocked(script, timeout, cancelled);
    }

    CommandResult Session::runLocked(const std::string& script, std::chrono::milliseconds timeout, const std::function<bool()>& cancelled) {
        const std::string id = randomHex(8);
        const std::string begin = "__SIRIUS_B_" + id, end = "__SIRIUS_E_" + id, exitMark = "__SIRIUS_X_" + id,
                          doc = "__SIRIUS_S_" + id;
        std::string text;
        text += "printf '\\n%s\\n' '" + begin + "'\n";
        text += "__sirius_f=$(mktemp 2>/dev/null || echo \"/tmp/sirius-$$-" + id + "\")\n";
        text += "__sirius_s=$(cat <<'" + doc + "'\n" + script + "\n" + doc + "\n)\n";
        text += "( eval \"$__sirius_s\" ) </dev/null 2>\"$__sirius_f\"\n";
        text += "__sirius_rc=$?\n";
        text += "printf '%s\\n' '" + end + "'\n";
        text += "cat \"$__sirius_f\" 2>/dev/null; rm -f \"$__sirius_f\"\n";
        text += "printf '%s %d\\n' '" + exitMark + "' \"$__sirius_rc\"\n";
        if (!child_ || !child_->writeInput(text)) throw SshError("the SSH connection is closed");
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        auto timedOut = [&] { return SshError("no answer from " + host_ + " within " + std::to_string(timeout.count() / 1000) + " s"); };
        std::string line;
        // anything before the begin marker (a login banner, a late answer) is not ours
        for (;;) {
            if (!readLine(line, deadline, cancelled)) throw timedOut();
            if (line == begin) break;
        }
        CommandResult r;
        bool first = true;
        for (;;) {
            if (!readLine(line, deadline, cancelled)) throw timedOut();
            if (endsWith(line, end)) {
                const std::string rest = line.substr(0, line.size() - end.size());
                if (!rest.empty()) r.out += (first ? "" : "\n") + rest;
                break;
            }
            r.out += (first ? "" : "\n") + line;
            first = false;
        }
        first = true;
        for (;;) {
            if (!readLine(line, deadline, cancelled)) throw timedOut();
            const std::size_t at = line.rfind(exitMark + " ");
            if (at != std::string::npos) {
                const std::string rest = line.substr(0, at);
                if (!rest.empty()) r.err += (first ? "" : "\n") + rest;
                r.exitCode = std::atoi(line.c_str() + at + exitMark.size() + 1);
                break;
            }
            r.err += (first ? "" : "\n") + line;
            first = false;
        }
        return r;
    }

} // namespace sirius::app::ssh
