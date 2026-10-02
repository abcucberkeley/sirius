#include "core/cluster.hpp"

#include "core/cancel.hpp"
#include "core/errors.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <map>
#include <sstream>

namespace sirius::app::cluster {

    using json = nlohmann::json;
    using ssh::shellQuote;
    using ssh::remotePathWord;

    // --- the profile -------------------------------------------------------------------

    json Profile::toJson() const {
        return {{"host", host}, {"checkout", checkout}, {"venv", venv}, {"partition", partition}, {"account", account}, {"qos", qos}, {"time", time}, {"gpus", gpus}, {"cpus", cpus}, {"mem", mem}, {"port", port}, {"ssh", sshProgram}};
    }

    Profile Profile::fromJson(const json& j) {
        Profile p;
        if (!j.is_object()) return p;
        auto str = [&](const char* k, std::string& out) {
            if (j.contains(k) && j[k].is_string()) out = j[k].get<std::string>();
        };
        auto num = [&](const char* k, int& out) {
            if (j.contains(k) && j[k].is_number_integer()) out = j[k].get<int>();
        };
        str("host", p.host);
        str("checkout", p.checkout);
        str("venv", p.venv);
        str("partition", p.partition);
        str("account", p.account);
        str("qos", p.qos);
        str("time", p.time);
        num("gpus", p.gpus);
        num("cpus", p.cpus);
        str("mem", p.mem);
        num("port", p.port);
        str("ssh", p.sshProgram);
        return p;
    }

    const char* stepTitle(Step s) {
        switch (s) {
            case Step::Login: return "SSH login";
            case Step::Checks: return "Checks on the cluster";
            case Step::Submit: return "Submit the worker job";
            case Step::Queue: return "Wait in the queue";
            case Step::Start: return "Worker starting on the node";
            case Step::Hello: return "Connect to the worker";
        }
        return "";
    }

    // --- listing -------------------------------------------------------------------------

    std::string listingScript(const std::string& path, int maxEntries) {
        const std::string word = path.empty() ? std::string("\"$HOME\"") : remotePathWord(path);
        return "P=$(command -v python3 || command -v python) || { echo '{\"error\": \"no python3 on this host to list folders "
               "with\"}'; exit 0; }\n"
               "\"$P\" - " +
               word + " " + std::to_string(maxEntries) + " <<'SIRIUS_LS'\n"
                                                         "import json, os, sys\n"
                                                         "def clean(s):\n"
                                                         "    try:\n"
                                                         "        s.encode('utf-8')\n"
                                                         "        return s\n"
                                                         "    except UnicodeEncodeError:\n"
                                                         "        return s.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')\n"
                                                         "p = os.path.abspath(sys.argv[1])\n"
                                                         "cap = int(sys.argv[2])\n"
                                                         "out = {'path': clean(p), 'home': clean(os.path.expanduser('~')), 'entries': [], 'truncated': False}\n"
                                                         "try:\n"
                                                         "    with os.scandir(p) as it:\n"
                                                         "        for e in it:\n"
                                                         "            if len(out['entries']) >= cap:\n"
                                                         "                out['truncated'] = True\n"
                                                         "                break\n"
                                                         "            try:\n"
                                                         "                link = e.is_symlink()\n"
                                                         "                d = e.is_dir()\n"
                                                         "            except OSError:\n"
                                                         "                link, d = True, False\n"
                                                         "            try:\n"
                                                         "                st = e.stat()\n"
                                                         "            except OSError:\n"
                                                         "                st = None\n"
                                                         "            out['entries'].append([clean(e.name), int(d), 0 if (d or st is None) else int(st.st_size),\n"
                                                         "                                   0 if st is None else int(st.st_mtime), int(link)])\n"
                                                         "except OSError as e:\n"
                                                         "    out = {'error': '%s: %s' % (clean(p), e.strerror or e)}\n"
                                                         "print(json.dumps(out))\n"
                                                         "SIRIUS_LS\n";
    }

    Listing parseListing(const std::string& line) {
        json j;
        try {
            j = json::parse(line);
        } catch (const json::exception&) {
            throw ssh::SshError("the cluster's answer to a folder listing was not readable", line.substr(0, 400));
        }
        if (j.contains("error")) throw ssh::SshError(j["error"].is_string() ? j["error"].get<std::string>() : "cannot list the folder");
        Listing l;
        l.path = j.value("path", std::string());
        l.home = j.value("home", std::string());
        l.truncated = j.value("truncated", false);
        if (j.contains("entries") && j["entries"].is_array())
            for (const json& e : j["entries"]) {
                if (!e.is_array() || e.size() < 5 || !e[0].is_string()) continue;
                Entry en;
                en.name = e[0].get<std::string>();
                en.dir = e[1].is_number() && e[1].get<int>() != 0;
                en.size = e[2].is_number() ? e[2].get<std::uint64_t>() : 0;
                en.mtime = e[3].is_number() ? e[3].get<double>() : 0.0;
                en.link = e[4].is_number() && e[4].get<int>() != 0;
                l.entries.push_back(std::move(en));
            }
        auto lower = [](std::string s) {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        std::stable_sort(l.entries.begin(), l.entries.end(), [&](const Entry& a, const Entry& b) {
            if (a.dir != b.dir) return a.dir;
            return lower(a.name) < lower(b.name);
        });
        return l;
    }

    // --- the session --------------------------------------------------------------------

    namespace {
        // A step that could not be done: what to tell the user.
        struct Failure {
            Step step;
            std::string reason;
            std::string remote;
            std::string fix;
        };

        std::map<std::string, std::string> keyValues(const std::string& text) {
            std::map<std::string, std::string> kv;
            std::istringstream in(text);
            std::string line;
            while (std::getline(in, line)) {
                const std::size_t eq = line.find('=');
                if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
            }
            return kv;
        }

        std::string trim(std::string s) {
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
            std::size_t i = 0;
            while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
            return s.substr(i);
        }

        std::string elapsed(std::chrono::steady_clock::duration d) {
            const long long s = std::chrono::duration_cast<std::chrono::seconds>(d).count();
            char buf[32];
            if (s >= 3600) std::snprintf(buf, sizeof buf, "%lld:%02lld:%02lld", s / 3600, (s / 60) % 60, s % 60);
            else std::snprintf(buf, sizeof buf, "%lld:%02lld", s / 60, s % 60);
            return buf;
        }
    } // namespace

    struct Session::Impl {
        mutable std::mutex m;
        Status status;
        Profile profile;
        PromptFn prompt;
        std::string askpassProgram;
        std::function<void()> changed;
        std::function<void(const std::string&)> log;
        std::chrono::milliseconds queuePoll{3000}, keepAlive{15000};

        std::shared_ptr<ssh::Session> ssh;            // under m
        std::unique_ptr<ssh::AskpassServer> askpass;
        std::string token;
        std::mutex controlMutex;
        std::unique_ptr<RemoteWorker> control;       // under controlMutex

        std::thread worker, keeper;
        std::atomic<bool> cancel{false}, stopKeeper{false}, connecting{false};
        std::atomic<bool> abortLogin{false}, loginActive{false};
        std::mutex keeperMutex;
        std::condition_variable keeperWake;

        // --- status ------------------------------------------------------------------
        void update(const std::function<void(Status&)>& fn) {
            {
                const std::lock_guard<std::mutex> g(m);
                fn(status);
            }
            if (changed) changed();
        }
        void say(const std::string& line) {
            if (log) log(line);
        }
        void stepState(Step s, StepStatus st, const std::string& detail) {
            update([&](Status& x) {
                x.steps[static_cast<std::size_t>(s)] = StepState{st, detail};
            });
        }
        void setState(State st, const std::string& reason = {}) {
            update([&](Status& x) {
                x.state = st;
                x.since = std::chrono::steady_clock::now();
                if (!reason.empty() || st != State::Disconnected) x.reason = reason;
            });
        }
        std::shared_ptr<ssh::Session> sshSession() const {
            const std::lock_guard<std::mutex> g(m);
            return ssh;
        }
        ssh::CommandResult remote(const std::string& script, std::chrono::milliseconds timeout = std::chrono::seconds(60)) {
            auto s = sshSession();
            if (!s || !s->isOpen()) throw ssh::SshError("the SSH connection is closed");
            return s->run(script, timeout, [this] { return cancel.load(); });
        }
        bool sleepCancellable(std::chrono::milliseconds d) {
            const auto end = std::chrono::steady_clock::now() + d;
            while (std::chrono::steady_clock::now() < end) {
                if (cancel.load()) return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return !cancel.load();
        }

        // --- askpass -----------------------------------------------------------------
        std::optional<std::string> ask(const ssh::Prompt& p) {
            if (p.notifyOnly) {
                say("Cluster: " + p.text);
                return std::string();
            }
            std::optional<std::string> answer = (prompt && loginActive.load()) ? prompt(p) : std::nullopt;
            if (!answer) {
                // Stop ssh before the helper answers: ssh would send an empty
                // response for a prompt the helper gives up on.
                abortLogin.store(true);
                for (int i = 0; i < 100 && loginActive.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return answer;
        }

        // --- the steps -----------------------------------------------------------------
        void login(const Profile& p) {
            stepState(Step::Login, StepStatus::Running, "ssh " + p.host);
            say("Cluster: logging in to " + p.host + "\xE2\x80\xA6");
            if (!askpass) askpass = std::make_unique<ssh::AskpassServer>([this](const ssh::Prompt& pr) { return ask(pr); });
            ssh::Options o;
            o.program = p.sshProgram;
            o.programArgs = p.sshProgramArgs;
            o.host = p.host;
            if (!askpassProgram.empty()) o.environment = askpass->environment(askpassProgram);
            auto s = std::make_shared<ssh::Session>();
            abortLogin.store(false);
            loginActive.store(true);
            try {
                s->open(o, [this] { return cancel.load() || abortLogin.load(); }, std::chrono::minutes(10));
            } catch (const ssh::SshError& e) {
                loginActive.store(false);
                if (abortLogin.load()) throw Failure{Step::Login, "Login cancelled: nothing was sent for the prompt you closed.", {}, {}};
                if (cancel.load()) throw Failure{Step::Login, "Cancelled.", {}, {}};
                throw Failure{Step::Login, std::string("SSH login to ") + p.host + " failed: " + e.what() + ".", e.detail,
                              "Check the host, user and password, then press Connect again (a failed login is never retried)."};
            }
            loginActive.store(false);
            {
                const std::lock_guard<std::mutex> g(m);
                ssh = s;
                status.sshUp = true;
                status.host = p.host;
            }
            stepState(Step::Login, StepStatus::Done, "logged in to " + p.host);
            say("Cluster: logged in to " + p.host);
        }

        void checks(const Profile& p) {
            stepState(Step::Checks, StepStatus::Running, "sbatch, the checkout, the Python environment");
            const std::string co = remotePathWord(p.checkout);
            std::string script;
            script += "command -v sbatch >/dev/null 2>&1 && echo sbatch=yes || echo sbatch=no\n";
            script += "[ -f " + co + "/app/python/sirius_worker/__main__.py ] && echo worker=yes || echo worker=no\n";
            script += "[ -f " + co + "/app/python/slurm/sirius_worker.sbatch ] && echo template=yes || echo template=no\n";
            script += "type module >/dev/null 2>&1 && module load python >/dev/null 2>&1\n";
            if (!p.venv.empty())
                script += "if [ -f " + remotePathWord(p.venv) + "/bin/activate ]; then echo venv=yes; . " + remotePathWord(p.venv) +
                          "/bin/activate; else echo venv=no; fi\n";
            script += "PY=$(command -v python3 || command -v python); echo \"python=$PY\"\n";
            script += "[ -n \"$PY\" ] && \"$PY\" -c 'import importlib.util as u, sys; print(\"pyver=%d.%d\" % sys.version_info[:2]); "
                      "[print(\"has_%s=%s\" % (m, \"yes\" if u.find_spec(m) else \"no\")) for m in (\"numpy\", \"tifffile\", \"torch\")]'\n";
            const ssh::CommandResult r = remote(script, std::chrono::seconds(90));
            auto kv = keyValues(r.out);
            const std::string pip = (p.venv.empty() ? std::string("python3 -m pip") : p.venv + "/bin/pip");
            const std::string req = p.checkout + "/app/python/requirements.txt";
            if (kv["sbatch"] != "yes")
                throw Failure{Step::Checks, "There is no sbatch on " + p.host + ": connect to the cluster's login node, the one you submit jobs from.", r.err, {}};
            if (kv["worker"] != "yes" || kv["template"] != "yes")
                throw Failure{Step::Checks, "There is no SIRIUS checkout at " + p.checkout + " on " + p.host + " (it needs app/python/sirius_worker and app/python/slurm).",
                              r.err, "Clone or copy this SIRIUS repository there (same version as this application), or set the checkout path."};
            if (!p.venv.empty() && kv["venv"] != "yes")
                throw Failure{Step::Checks, "There is no Python environment at " + p.venv + " on " + p.host + ".", r.err,
                              "python3 -m venv " + p.venv + " && " + p.venv + "/bin/pip install -r " + req + " tifffile"};
            if (kv["python"].empty()) throw Failure{Step::Checks, "There is no python3 on " + p.host + ".", r.err, "module load python, or set a venv"};
            if (kv["has_numpy"] != "yes")
                throw Failure{Step::Checks, "numpy is missing from " + (p.venv.empty() ? kv["python"] : p.venv) + ": the worker cannot start without it.", r.err,
                              pip + " install -r " + req};
            std::string detail = "sbatch \xC2\xB7 checkout \xC2\xB7 python " + kv["pyver"] + " \xC2\xB7 numpy";
            detail += kv["has_torch"] == "yes" ? " \xC2\xB7 torch" : " \xC2\xB7 no torch (models will not run)";
            if (kv["has_tifffile"] != "yes") {
                update([&](Status& x) { x.fix = pip + " install tifffile"; });
                stepState(Step::Checks, StepStatus::Warning, detail + " \xC2\xB7 no tifffile: cluster datasets cannot be opened");
                say("Cluster: tifffile is missing on " + p.host + " (" + pip + " install tifffile)");
            } else {
                stepState(Step::Checks, StepStatus::Done, detail + " \xC2\xB7 tifffile");
            }
        }

        void submit(const Profile& p) {
            stepState(Step::Submit, StepStatus::Running, "sbatch");
            token = ssh::randomHex(16);
            std::string script = "cd " + remotePathWord(p.checkout) + " || exit 3\n";
            // the token reaches the job through this shell's environment (sbatch exports it); never an argument
            script += "export SIRIUS_TOKEN=" + shellQuote(token) + "\n";
            script += "export SIRIUS_PORT=" + std::to_string(p.port) + "\nexport SIRIUS_MAX_CLIENTS=8\n";
            if (!p.venv.empty()) script += "export SIRIUS_VENV=" + remotePathWord(p.venv) + "\n";
            if (p.gpus <= 0) script += "export SIRIUS_DEVICE=cpu\n";
            std::string cmd = "sbatch --parsable --job-name=sirius-worker --output=sirius-worker-%j.log";
            if (!p.partition.empty()) cmd += " --partition=" + shellQuote(p.partition);
            if (!p.account.empty()) cmd += " --account=" + shellQuote(p.account);
            if (!p.qos.empty()) cmd += " --qos=" + shellQuote(p.qos);
            if (!p.time.empty()) cmd += " --time=" + shellQuote(p.time);
            cmd += p.gpus > 0 ? " --gres=gpu:" + std::to_string(p.gpus) : std::string(" --gres=none");
            if (p.cpus > 0) cmd += " --cpus-per-task=" + std::to_string(p.cpus);
            if (!p.mem.empty()) cmd += " --mem=" + shellQuote(p.mem);
            script += cmd + " app/python/slurm/sirius_worker.sbatch\n";
            const ssh::CommandResult r = remote(script, std::chrono::seconds(60));
            std::string id;
            for (char c : trim(r.out)) {
                if (std::isdigit(static_cast<unsigned char>(c))) id.push_back(c);
                else if (!id.empty()) break;
            }
            if (!r.ok() || id.empty())
                throw Failure{Step::Submit, "sbatch refused the worker job.", trim(r.err.empty() ? r.out : r.err),
                              "Check the partition, account, QoS and time in the profile."};
            update([&](Status& x) { x.jobId = id; });
            std::string where = p.partition.empty() ? std::string() : " to " + p.partition;
            stepState(Step::Submit, StepStatus::Done, "job " + id + where);
            say("Cluster: submitted job " + id + where);
        }

        std::string jobLog(const Profile& p, const std::string& id) {
            try {
                return trim(remote("tail -n 30 " + remotePathWord(p.checkout) + "/sirius-worker-" + id + ".log 2>/dev/null", std::chrono::seconds(30)).out);
            } catch (const ssh::SshError&) {
                return {};
            }
        }

        std::string finalState(const std::string& id) {
            try {
                const ssh::CommandResult r = remote("sacct -n -X -P -j " + id + " -o State,ExitCode 2>/dev/null | head -n 1", std::chrono::seconds(30));
                std::string s = trim(r.out);
                const std::size_t bar = s.find('|');
                if (bar != std::string::npos) s = s.substr(0, bar) + " (exit " + s.substr(bar + 1) + ")";
                return s.empty() ? std::string("no longer in the queue") : s;
            } catch (const ssh::SshError&) {
                return "no longer in the queue";
            }
        }

        std::string waitInQueue(const Profile& p, const std::string& id) {
            stepState(Step::Queue, StepStatus::Running, "job " + id);
            const auto t0 = std::chrono::steady_clock::now();
            for (;;) {
                const ssh::CommandResult r = remote("squeue -h -j " + id + " -o '%T|%r|%N' 2>/dev/null", std::chrono::seconds(30));
                const std::string line = trim(r.out);
                if (line.empty()) {
                    const std::string fin = finalState(id);
                    throw Failure{Step::Queue, "Job " + id + " ended before the worker ran: " + fin + ".", jobLog(p, id), {}};
                }
                std::string state = line, reason, node;
                std::size_t a = line.find('|');
                if (a != std::string::npos) {
                    state = line.substr(0, a);
                    const std::size_t b = line.find('|', a + 1);
                    reason = line.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1);
                    if (b != std::string::npos) node = line.substr(b + 1);
                }
                update([&](Status& x) { x.jobState = state; });
                if (state == "RUNNING" && !node.empty() && node != "(null)") {
                    stepState(Step::Queue, StepStatus::Done, "job " + id + " \xC2\xB7 waited " + elapsed(std::chrono::steady_clock::now() - t0));
                    return node;
                }
                std::string detail = "job " + id + " \xC2\xB7 " + state;
                if (!reason.empty() && reason != "None") detail += " (" + reason + ")";
                detail += " \xC2\xB7 " + elapsed(std::chrono::steady_clock::now() - t0);
                stepState(Step::Queue, StepStatus::Running, detail);
                if (!sleepCancellable(queuePoll)) throw Failure{Step::Queue, "Cancelled while job " + id + " waits in the queue.", {}, {}};
            }
        }

        void waitForWorker(const Profile& p, const std::string& id, const std::string& node) {
            stepState(Step::Start, StepStatus::Running, "on " + node);
            update([&](Status& x) { x.node = node; });
            say("Cluster: job " + id + " runs on " + node);
            const auto t0 = std::chrono::steady_clock::now();
            const std::string logFile = remotePathWord(p.checkout) + "/sirius-worker-" + id + ".log";
            for (;;) {
                const std::string script = "echo state=$(squeue -h -j " + id + " -o %T 2>/dev/null)\n[ -f " + logFile + " ] && grep -m1 -E '^\\{\"(port|error)\"' " +
                                           logFile + " | sed 's/^/announce=/'\ntrue\n";
                const ssh::CommandResult r = remote(script, std::chrono::seconds(30));
                auto kv = keyValues(r.out);
                const std::string announce = kv["announce"];
                if (!announce.empty()) {
                    json j;
                    try {
                        j = json::parse(announce);
                    } catch (const json::exception&) {
                    }
                    if (j.contains("error")) {
                        std::string missing;
                        if (j.contains("missing") && j["missing"].is_array())
                            for (const json& x : j["missing"]) missing += (missing.empty() ? "" : ", ") + x.get<std::string>();
                        throw Failure{Step::Start, "The worker on " + node + " cannot start: " + (missing.empty() ? std::string("a package is missing") : missing + " missing") + ".",
                                      jobLog(p, id), (p.venv.empty() ? std::string("python3 -m pip") : p.venv + "/bin/pip") + " install -r " + p.checkout + "/app/python/requirements.txt"};
                    }
                    if (j.contains("port")) {
                        stepState(Step::Start, StepStatus::Done, "listening on " + node + ":" + std::to_string(p.port));
                        return;
                    }
                }
                if (trim(kv["state"]).empty() || (kv["state"] != "RUNNING" && kv["state"] != "COMPLETING")) {
                    const std::string fin = finalState(id);
                    throw Failure{Step::Start, "Job " + id + " ended before the worker listened: " + fin + ".", jobLog(p, id), {}};
                }
                stepState(Step::Start, StepStatus::Running, "on " + node + " \xC2\xB7 starting \xC2\xB7 " + elapsed(std::chrono::steady_clock::now() - t0));
                if (!sleepCancellable(queuePoll)) throw Failure{Step::Start, "Cancelled while the worker starts.", {}, {}};
            }
        }

        void hello(const Profile& p, const std::string& node) {
            stepState(Step::Hello, StepStatus::Running, node + ":" + std::to_string(p.port) + " through the SSH tunnel");
            int socks = 0;
            {
                auto s = sshSession();
                socks = s ? s->socksPort() : 0;
            }
            std::unique_ptr<RemoteWorker> w;
            try {
                w = RemoteWorker::connect(node, p.port, token, std::chrono::seconds(20), [this] { return cancel.load(); }, socks);
            } catch (const CancelledError&) {
                throw Failure{Step::Hello, "Cancelled while the worker answers.", {}, {}};
            } catch (const std::exception& e) {
                throw Failure{Step::Hello, "Could not reach the worker on " + node + ":" + std::to_string(p.port) + " through the SSH tunnel.", e.what(), {}};
            }
            w->setCancelGrace(std::chrono::milliseconds(0));
            const WorkerCapabilities caps = w->capabilities();
            int kinds = 0;
            for (const std::string& meth : caps.methods)
                if (meth.rfind("run:", 0) == 0) ++kinds;
            {
                const std::lock_guard<std::mutex> g(controlMutex);
                control = std::move(w);
            }
            update([&](Status& x) { x.caps = caps; });
            stepState(Step::Hello, StepStatus::Done,
                      "sirius_worker " + caps.version + " \xC2\xB7 " + caps.device + " \xC2\xB7 " + std::to_string(kinds) + " step kinds");
        }

        void run() {
            Profile p;
            {
                const std::lock_guard<std::mutex> g(m);
                p = profile;
            }
            try {
                auto s = sshSession();
                const bool reuse = s && s->isOpen() && s->host() == p.host;
                if (reuse) stepState(Step::Login, StepStatus::Done, "logged in to " + p.host);
                else {
                    if (s) s->close();
                    login(p);
                }
                checks(p);
                submit(p);
                std::string id;
                {
                    const std::lock_guard<std::mutex> g(m);
                    id = status.jobId;
                }
                const std::string node = waitInQueue(p, id);
                waitForWorker(p, id, node);
                hello(p, node);
                setState(State::Connected);
                Status st;
                {
                    const std::lock_guard<std::mutex> g(m);
                    st = status;
                }
                say("HPC: connected to the worker on " + st.node + " (job " + st.jobId + ", " + st.caps.device + ")");
                startKeeper();
            } catch (const Failure& f) {
                update([&](Status& x) {
                    x.steps[static_cast<std::size_t>(f.step)] = StepState{StepStatus::Failed, f.reason};
                    x.state = State::Disconnected;
                    x.since = std::chrono::steady_clock::now();
                    x.reason = f.reason;
                    x.remoteOutput = f.remote;
                    if (!f.fix.empty()) x.fix = f.fix;
                    auto s2 = ssh;
                    x.sshUp = s2 && s2->isOpen();
                });
                say("HPC: " + f.reason + (f.remote.empty() ? std::string() : " \xE2\x80\x94 " + f.remote.substr(0, 300)));
            } catch (const std::exception& e) {
                update([&](Status& x) {
                    x.state = State::Disconnected;
                    x.since = std::chrono::steady_clock::now();
                    x.reason = cancel.load() ? std::string("Cancelled.") : std::string(e.what());
                    for (StepState& st : x.steps)
                        if (st.status == StepStatus::Running) st = StepState{StepStatus::Failed, x.reason};
                    auto s2 = ssh;
                    x.sshUp = s2 && s2->isOpen();
                });
                say(std::string("HPC: ") + (cancel.load() ? "cancelled" : e.what()));
            }
            connecting.store(false);
        }

        // --- keep-alive ------------------------------------------------------------------
        void startKeeper() {
            stopKeeper.store(false);
            if (keeper.joinable()) keeper.join();
            keeper = std::thread([this] { keep(); });
        }

        void lost(const std::string& reason, const std::string& remoteText = {}) {
            {
                const std::lock_guard<std::mutex> g(controlMutex);
                control.reset();
            }
            update([&](Status& x) {
                x.state = State::Disconnected;
                x.since = std::chrono::steady_clock::now();
                x.reason = reason;
                x.remoteOutput = remoteText;
                auto s = ssh;
                x.sshUp = s && s->isOpen();
            });
            say("HPC: disconnected: " + reason);
        }

        void keep() {
            int tick = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(keeperMutex);
                    if (keeperWake.wait_for(lk, keepAlive, [this] { return stopKeeper.load(); })) return;
                }
                ++tick;
                Profile p;
                std::string id, host;
                {
                    const std::lock_guard<std::mutex> g(m);
                    p = profile;
                    id = status.jobId;
                    host = status.host;
                }
                auto s = sshSession();
                if (!s || !s->isOpen()) {
                    lost("the SSH connection to " + host + " ended", s ? s->stderrTail() : std::string());
                    return;
                }
                bool pingFailed = false;
                std::string why;
                {
                    const std::lock_guard<std::mutex> g(controlMutex);
                    if (!control) return;
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                    try {
                        control->call("ping", json::object(), {}, {}, [&] { return stopKeeper.load() || std::chrono::steady_clock::now() > deadline; });
                    } catch (const std::exception& e) {
                        if (stopKeeper.load()) return;
                        pingFailed = true;
                        why = isCancellation(e) ? std::string("no answer within 20 s") : std::string(e.what());
                    }
                }
                if (pingFailed || tick % 2 == 0) {
                    std::string state;
                    try {
                        state = trim(remote("squeue -h -j " + id + " -o %T 2>/dev/null", std::chrono::seconds(30)).out);
                    } catch (const std::exception&) {
                        state = "?";
                    }
                    if (state.empty() || (state != "RUNNING" && state != "COMPLETING" && state != "?")) {
                        lost("job " + id + " ended: " + finalState(id), jobLog(p, id));
                        return;
                    }
                    if (pingFailed) {
                        lost("the worker stopped answering (" + why + ")", jobLog(p, id));
                        return;
                    }
                }
            }
        }

        void stopKeeperThread() {
            stopKeeper.store(true);
            keeperWake.notify_all();
            if (keeper.joinable() && keeper.get_id() != std::this_thread::get_id()) keeper.join();
        }

        void stopWorkerThread() {
            cancel.store(true);
            abortLogin.store(true);
            if (worker.joinable()) worker.join();
            cancel.store(false);
        }
    };

    Session::Session() : impl_(std::make_unique<Impl>()) {}

    Session::~Session() {
        impl_->stopKeeperThread();
        impl_->stopWorkerThread();
        {
            const std::lock_guard<std::mutex> g(impl_->controlMutex);
            impl_->control.reset();
        }
        auto s = impl_->sshSession();
        if (s) s->close();
    }

    void Session::setPrompt(PromptFn fn) { impl_->prompt = std::move(fn); }
    void Session::setAskpassProgram(std::string program) { impl_->askpassProgram = std::move(program); }
    void Session::setChanged(std::function<void()> fn) { impl_->changed = std::move(fn); }
    void Session::setLog(std::function<void(const std::string&)> fn) { impl_->log = std::move(fn); }
    void Session::setPollInterval(std::chrono::milliseconds queue, std::chrono::milliseconds keepAlive) {
        impl_->queuePoll = queue;
        impl_->keepAlive = keepAlive;
    }

    void Session::connect(const Profile& profile) {
        if (impl_->connecting.exchange(true)) return;
        impl_->stopKeeperThread();
        if (impl_->worker.joinable()) impl_->worker.join();
        {
            const std::lock_guard<std::mutex> g(impl_->controlMutex);
            impl_->control.reset();
        }
        impl_->cancel.store(false);
        impl_->update([&](Status& x) {
            const bool sshUp = x.sshUp;
            const std::string host = x.host;
            x = Status{};
            x.state = State::Connecting;
            x.since = std::chrono::steady_clock::now();
            x.sshUp = sshUp;
            x.host = host;
        });
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            impl_->profile = profile;
        }
        impl_->worker = std::thread([this] { impl_->run(); });
    }

    void Session::cancelConnect() {
        impl_->cancel.store(true);
        impl_->abortLogin.store(true);
    }

    void Session::disconnect(bool cancelJob) {
        impl_->stopKeeperThread();
        impl_->stopWorkerThread();
        {
            const std::lock_guard<std::mutex> g(impl_->controlMutex);
            impl_->control.reset();
        }
        std::string id, note;
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            id = impl_->status.jobId;
        }
        auto s = impl_->sshSession();
        if (!id.empty()) {
            if (cancelJob && s && s->isOpen()) {
                try {
                    const ssh::CommandResult r = s->run("scancel " + id, std::chrono::seconds(30));
                    note = r.ok() ? "job " + id + " cancelled" : "scancel " + id + " failed: " + trim(r.err);
                    if (r.ok()) {
                        const std::lock_guard<std::mutex> g(impl_->m);
                        impl_->status.jobId.clear();
                    }
                } catch (const std::exception& e) {
                    note = "scancel " + id + " failed: " + e.what();
                }
            } else {
                note = "job " + id + " left running";
            }
        }
        if (s) s->close();
        impl_->update([&](Status& x) {
            x.state = State::Disconnected;
            x.since = std::chrono::steady_clock::now();
            x.reason = "disconnected" + (note.empty() ? std::string() : " (" + note + ")");
            x.sshUp = false;
        });
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            impl_->ssh.reset();
        }
        impl_->say("HPC: disconnected" + (note.empty() ? std::string() : ": " + note));
    }

    Status Session::status() const {
        const std::lock_guard<std::mutex> g(impl_->m);
        Status s = impl_->status;
        if (impl_->ssh) s.sshUp = impl_->ssh->isOpen();
        return s;
    }

    Profile Session::profile() const {
        const std::lock_guard<std::mutex> g(impl_->m);
        return impl_->profile;
    }

    bool Session::connected() const { return status().state == State::Connected; }
    bool Session::sshUp() const { return status().sshUp; }

    bool Session::hasJob() const {
        const std::lock_guard<std::mutex> g(impl_->m);
        return !impl_->status.jobId.empty();
    }

    Session::Endpoint Session::endpoint() const {
        const std::lock_guard<std::mutex> g(impl_->m);
        Endpoint e;
        if (impl_->status.state != State::Connected) return e;
        e.host = impl_->status.node;
        e.port = impl_->profile.port;
        e.token = impl_->token;
        e.socksPort = impl_->ssh ? impl_->ssh->socksPort() : 0;
        return e;
    }

    std::unique_ptr<RemoteWorker> Session::connectWorker(std::chrono::milliseconds timeout, const std::function<bool()>& cancelled) const {
        const Endpoint e = endpoint();
        if (e.host.empty()) throw ProtocolError("not connected to a cluster worker (Process \xE2\x96\xB8 Connect to cluster\xE2\x80\xA6)");
        return RemoteWorker::connect(e.host, e.port, e.token, timeout, cancelled, e.socksPort);
    }

    Listing Session::list(const std::string& path, int maxEntries) {
        auto s = impl_->sshSession();
        if (!s || !s->isOpen()) throw ssh::SshError("not logged in to the cluster: connect first");
        const ssh::CommandResult r = s->run(listingScript(path, maxEntries), std::chrono::seconds(60));
        std::string line;
        std::istringstream in(r.out);
        std::string l;
        while (std::getline(in, l))
            if (!l.empty() && l.front() == '{') line = l;
        if (line.empty()) throw ssh::SshError("the cluster did not list " + path, trim(r.err));
        return parseListing(line);
    }

    ssh::CommandResult Session::run(const std::string& script, std::chrono::milliseconds timeout) {
        auto s = impl_->sshSession();
        if (!s || !s->isOpen()) throw ssh::SshError("not logged in to the cluster: connect first");
        return s->run(script, timeout);
    }

} // namespace sirius::app::cluster
