// `sirius-cli serve`: SIRIUS's engine as the HPC backend's worker
// (core/engine_server.hpp). It listens, prints the one announce line the
// cluster job's log is read for -- as `python -m sirius_worker` does -- and
// serves until a client's shutdown, SIGTERM or Ctrl+C. Nothing else goes to
// stdout; the log goes to stderr.

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <fstream>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "cli/args.hpp"
#include "cli/commands.hpp"
#include "cli/stdio.hpp"
#include "core/engine_server.hpp"
#include "core/host.hpp"
#include "core/rpc.hpp"
#include "core/secure_wipe.hpp"

namespace sirius::cli {

    namespace app = sirius::app;

    namespace {

        // The engine the signal handlers and the stdin reader stop; null
        // outside serve (they may run after it has returned).
        std::mutex activeMutex;
        app::EngineServer* active = nullptr;

        void stopActive() {
            const std::lock_guard<std::mutex> g(activeMutex);
            if (active) active->stop();
        }

        void unsetEnvironment(const char* name) {
#ifdef _WIN32
            _putenv_s(name, "");
#else
            unsetenv(name);
#endif
        }

        // The token in `path`, which is then deleted. On POSIX the file must be
        // a regular file of this user's that nobody else can read or write
        // (sirius_worker/__main__.py: read_token_file).
        std::string readTokenFile(const std::string& path) {
            std::string data;
#ifdef _WIN32
            {
                std::ifstream in(path, std::ios::binary);
                if (!in) throw UsageError("cannot read the token file " + path);
                char buffer[4096];
                in.read(buffer, sizeof buffer);
                data.assign(buffer, static_cast<std::size_t>(in.gcount()));
            }
#else
            const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW);
            if (fd < 0) throw UsageError("cannot read the token file " + path + ": " + std::strerror(errno));
            struct stat st{};
            if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
                ::close(fd);
                throw UsageError("the token file " + path + " is not a regular file");
            }
            if (st.st_uid != ::getuid()) {
                ::close(fd);
                throw UsageError("the token file " + path + " belongs to another user");
            }
            if (st.st_mode & 077) {
                ::close(fd);
                throw UsageError("the token file " + path + " can be read or written by others: create it with umask 077, or chmod 600 it");
            }
            char buffer[4096];
            const ssize_t n = ::read(fd, buffer, sizeof buffer);
            ::close(fd);
            if (n > 0) data.assign(buffer, static_cast<std::size_t>(n));
#endif
            if (std::remove(path.c_str()) != 0) writeError("sirius-cli serve: could not delete the token file " + path + "\n");
            std::size_t b = 0, e = data.size();
            while (b < e && std::isspace(static_cast<unsigned char>(data[b]))) ++b;
            while (e > b && std::isspace(static_cast<unsigned char>(data[e - 1]))) --e;
            std::string token = data.substr(b, e - b);
            app::secureWipe(data);
            if (token.empty()) throw UsageError("the token file " + path + " is empty");
            return token;
        }

        long long integer(const Args& args, const char* name, long long def, long long lo, long long hi) {
            const std::string v = args.value(name);
            if (v.empty()) return def;
            char* end = nullptr;
            const long long x = std::strtoll(v.c_str(), &end, 10);
            if (!end || *end != '\0' || x < lo || x > hi)
                throw UsageError("--" + std::string(name) + " expects a whole number from " + std::to_string(lo) + " to " + std::to_string(hi) + ", not '" + v + "'");
            return x;
        }

        std::string device(const std::string& text) {
            std::string v;
            for (char c : text) v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            if (v.empty()) return "auto";
            if (v == "auto" || v == "cpu" || v == "cuda") return v;
            if (v.rfind("cuda:", 0) == 0 && v.size() > 5 && v.size() < 9 && v.find_first_not_of("0123456789", 5) == std::string::npos) return v;
            throw UsageError("--device must be auto, cpu, cuda or cuda:N, not '" + text + "'");
        }

        int serveImpl(const Args& args) {
            const GlobalOptions& g = args.global;
            app::EngineOptions o;
            o.host = args.value("host", "127.0.0.1");
            const int port = static_cast<int>(integer(args, "port", 0, 0, 65535));
            o.maxClients = static_cast<int>(integer(args, "max-clients", 8, 1, 1024));
            o.device = device(args.value("device", "auto"));
            if (args.has("idle-timeout")) {
                const std::string v = args.value("idle-timeout");
                char* end = nullptr;
                const double s = std::strtod(v.c_str(), &end);
                if (!end || *end != '\0' || !(s >= 0.0) || s > 1e9) throw UsageError("--idle-timeout expects seconds (0 = never), not '" + v + "'");
                o.idleTimeout = std::chrono::milliseconds(static_cast<long long>(s * 1000.0));
            }
            o.pythonWorker = !args.has("no-python-worker");
            o.python = g.python;
            o.workerDir = g.workerDir;
            o.scratch = g.scratch;
            const bool quiet = g.quiet;
            o.log = [quiet](const std::string& line) {
                if (!quiet) writeError("sirius-cli serve: " + line + "\n");
            };

            // The token: a file (read, then deleted), else $SIRIUS_TOKEN. Both
            // variables leave the environment, so no child inherits them.
            const std::string fileOption = args.value("token-file");
            const std::string envFile = app::host::environment("SIRIUS_TOKEN_FILE");
            std::string envToken = app::host::environment("SIRIUS_TOKEN");
            unsetEnvironment("SIRIUS_TOKEN_FILE");
            unsetEnvironment("SIRIUS_TOKEN");
            const std::string tokenFile = !fileOption.empty() ? fileOption : envFile;
            o.token = !tokenFile.empty() ? readTokenFile(tokenFile) : envToken;
            app::secureWipe(envToken);
            if (o.token.empty() && !app::rpc::isLoopbackHost(o.host))
                throw UsageError("refusing to listen on " + (o.host.empty() ? std::string("0.0.0.0") : o.host) +
                                     " without a token: any host that can reach this port could run code as this user",
                                 "set $SIRIUS_TOKEN_FILE (or --token-file, or $SIRIUS_TOKEN), or bind 127.0.0.1 and reach it through an SSH tunnel");
            if (o.token.empty()) o.log("no token: every client that can connect is served (loopback only)");

            std::unique_ptr<app::rpc::Listener> listener;
            try {
                listener = std::make_unique<app::rpc::Listener>(o.host, port);
            } catch (const std::exception& e) {
                throw UsageError(e.what());
            }
            app::EngineServer engine(o);
            app::secureWipe(o.token);
            {
                const std::lock_guard<std::mutex> lk(activeMutex);
                active = &engine;
            }
            installInterruptHandler([] { stopActive(); });
            installTerminationHandler([] { stopActive(); });
            if (args.has("exit-with-parent")) startStdinReader([](std::string) {}, [] { stopActive(); });
            // the one line the launcher reads (cluster.cpp waits for ^{"port")
            writeLine(engine.announce(listener->port()).dump());
            engine.serve(*listener);
            {
                const std::lock_guard<std::mutex> lk(activeMutex);
                active = nullptr;
            }
            o.log("stopped");
            return 0;
        }

    } // namespace

    int serveEngine(const Args& args) {
        try {
            return serveImpl(args);
        } catch (const UsageError& e) {
            writeError("sirius-cli: usage: " + std::string(e.what()) + (e.hint().empty() ? std::string() : " (" + e.hint() + ")") + "\n");
            return 2;
        } catch (const std::exception& e) {
            writeError("sirius-cli: internal: " + std::string(e.what()) + "\n");
            return 1;
        }
    }

} // namespace sirius::cli
