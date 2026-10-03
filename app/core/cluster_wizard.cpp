#include "core/cluster_wizard.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sirius::app::cluster::wizard {

    namespace {

        const std::string kDot = " \xC2\xB7 ";

        std::string lower(std::string s) {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        bool has(const std::string& haystack, const char* needle) { return haystack.find(needle) != std::string::npos; }

        const StepState& step(const Status& st, Step s) { return st.steps[static_cast<std::size_t>(s)]; }

        bool workerStepFailed(const Status& st) {
            for (const Step s : {Step::Checks, Step::Start, Step::Hello})
                if (step(st, s).status == StepStatus::Failed) return true;
            return false;
        }

        // "job 4711 · PENDING (Resources) · 0:42" -> its parts
        std::vector<std::string> dotted(const std::string& s) {
            std::vector<std::string> out;
            std::size_t from = 0;
            for (;;) {
                const std::size_t at = s.find(kDot, from);
                out.push_back(s.substr(from, at == std::string::npos ? std::string::npos : at - from));
                if (at == std::string::npos) break;
                from = at + kDot.size();
            }
            return out;
        }

        std::string jsonString(const nlohmann::json& j, const char* key) {
            if (!j.is_object()) return {};
            const auto it = j.find(key);
            return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
        }

    } // namespace

    bool loggedIn(const Status& st, const std::string& host) { return st.sshUp && !host.empty() && st.host == host; }

    bool jobHeld(const Status& st) { return st.state == State::JobReady || st.state == State::Starting || st.state == State::Connected; }

    bool busy(const Status& st) { return st.state == State::Connecting || st.state == State::Starting; }

    Gate connectGate(const Status& st, const std::string& host) {
        if (host.empty()) return {false, "Type the cluster's name first"};
        if (busy(st)) return {false, "Busy: wait until it is done, or press Stop"};
        if (jobHeld(st))
            return {false, "A job is held on " + st.host + ": disconnect first (\xE2\x8B\xAF \xE2\x96\xB8 Disconnect\xE2\x80\xA6) to use another cluster"};
        return {true, {}};
    }

    Gate startJobGate(const Status& st, const std::string& host) {
        if (jobHeld(st)) return {false, "A job already holds a node: Change job\xE2\x80\xA6 cancels it first"};
        if (busy(st)) return {false, "Waiting for the job\xE2\x80\xA6"};
        if (!loggedIn(st, host)) return {false, "Log in first (page 1)"};
        return {true, {}};
    }

    Gate startWorkerGate(const Status& st, const std::string& image) {
        if (busy(st)) return {false, "The worker is starting\xE2\x80\xA6"};
        if (!jobHeld(st)) return {false, "Start the job first (page 2): the worker runs in it"};
        if (image.empty()) return {false, "Pick the worker image (.sif) on page 2 first"};
        return {true, {}};
    }

    Gate nextGate(Page page, const Status& st, const std::string& host) {
        switch (page) {
            case Page::Connect:
                if (st.state == State::Connecting && step(st, Step::Login).status == StepStatus::Running) return {false, "Logging in\xE2\x80\xA6"};
                if (loggedIn(st, host)) return {true, {}};
                return {false, "Connect first: Next is enabled once you are logged in"};
            case Page::Job:
                if (jobHeld(st) && st.host == host) return {true, {}};
                if (st.state == State::Connecting) return {false, "Waiting for the job to start on a node\xE2\x80\xA6"};
                return {false, "Start the job first: Next is enabled once Slurm gives it a node"};
            case Page::Worker:
                if (st.state == State::Connected) return {true, {}};
                if (st.state == State::Starting) return {false, "The worker is starting\xE2\x80\xA6"};
                if (st.state == State::JobReady && workerStepFailed(st)) return {false, "The worker did not start: see the report"};
                return {false, "Start the worker first: Finish is enabled once it answers"};
            case Page::Summary: return {true, {}};
        }
        return {};
    }

    bool pageDone(Page page, const Status& st, const std::string& host) {
        switch (page) {
            case Page::Connect: return loggedIn(st, host);
            case Page::Job: return jobHeld(st) && st.host == host;
            case Page::Worker: return st.state == State::Connected;
            case Page::Summary: return false;
        }
        return false;
    }

    Page openingPage(const Status& st, const std::string& host) {
        switch (st.state) {
            case State::Connected: return Page::Summary;
            case State::Starting: return Page::Worker;
            case State::JobReady: return workerStepFailed(st) ? Page::Worker : Page::Job;
            case State::Connecting: return step(st, Step::Login).status == StepStatus::Running ? Page::Connect : Page::Job;
            case State::Disconnected:
                if (step(st, Step::Login).status == StepStatus::Failed || !loggedIn(st, host)) return Page::Connect;
                return Page::Job;
            case State::Idle: return Page::Connect;
        }
        return Page::Connect;
    }

    // --- the login ------------------------------------------------------------------------

    std::string loginFailureWords(const std::string& host, const std::string& reason, const std::string& sshOutput) {
        const std::string all = lower(sshOutput + "\n" + reason);
        const std::string h = host.empty() ? std::string("the cluster") : host;
        if (has(all, "cancelled")) return reason.empty() ? std::string("The login was cancelled.") : reason;
        if (has(all, "permission denied") || has(all, "authentication failed"))
            return "Wrong password or code: " + h + " did not accept the login. Check it and press Connect again.";
        if (has(all, "could not resolve hostname") || has(all, "name or service not known") || has(all, "no such host is known") ||
            has(all, "nodename nor servname") || has(all, "temporary failure in name resolution"))
            return "Host not found: \"" + h + "\" is neither a Host of your ~/.ssh/config nor a name the network knows. Check the spelling.";
        if (has(all, "timed out") || has(all, "no answer from"))
            return "No answer from " + h + " (timed out): check your network connection or VPN.";
        if (has(all, "connection refused")) return h + " refused the connection: no SSH server answers there on that port.";
        if (has(all, "no route to host") || has(all, "network is unreachable"))
            return "The network cannot reach " + h + ": are you on the right network or VPN?";
        if (has(all, "host key verification failed") || has(all, "remote host identification has changed"))
            return h + "'s host key is not the one ~/.ssh/known_hosts has for it: ask your cluster's support before you go on.";
        if (has(all, "too many authentication failures"))
            return "Too many keys were offered: " + h + " stopped the login. Name one key (IdentityFile) for it in ~/.ssh/config.";
        if (has(all, "connection closed") || has(all, "connection reset") || has(all, "the ssh connection ended"))
            return h + " closed the connection during the login.";
        if (has(all, "no ssh client found")) return reason;
        return reason.empty() ? "The login to " + h + " failed." : reason;
    }

    LoginOutcome loginOutcome(const Status& st, const std::string& host, const std::string& user) {
        LoginOutcome o;
        if (st.state == State::Connecting && step(st, Step::Login).status == StepStatus::Running) {
            o.kind = LoginOutcome::Kind::Busy;
            o.text = "Logging in to " + (host.empty() ? std::string("the cluster") : host) + "\xE2\x80\xA6";
            return o;
        }
        if (loggedIn(st, host)) {
            o.kind = LoginOutcome::Kind::Ok;
            o.text = "Connected to " + host + (user.empty() ? std::string() : " as " + user);
            return o;
        }
        if (st.state == State::Disconnected && step(st, Step::Login).status == StepStatus::Failed) {
            o.kind = LoginOutcome::Kind::Failed;
            o.text = loginFailureWords(host, st.reason, st.remoteOutput);
            o.details = st.remoteOutput.empty() ? st.reason : st.remoteOutput;
            return o;
        }
        if (st.state == State::Disconnected && st.dropped && st.host == host) {
            o.kind = LoginOutcome::Kind::Failed;
            o.text = st.reason.empty() ? "The connection to " + host + " was lost." : st.reason;
            o.details = st.remoteOutput;
        }
        return o;
    }

    // --- the job ----------------------------------------------------------------------------

    std::string timeLeftText(const Status& st, std::chrono::steady_clock::time_point now) {
        if (st.jobLimitSeconds == -1) return "no time limit";
        if (st.jobLimitSeconds < 0 || st.jobStarted == std::chrono::steady_clock::time_point{}) return {};
        const long long used = std::chrono::duration_cast<std::chrono::seconds>(now - st.jobStarted).count();
        return durationText(std::max(0LL, st.jobLimitSeconds - used)) + " left";
    }

    JobLine jobLine(const Status& st, std::chrono::steady_clock::time_point now) {
        JobLine l;
        if (jobHeld(st)) {
            l.kind = JobLine::Kind::Running;
            l.text = "Running on " + st.node + kDot + "job " + st.jobId;
            if (const std::string left = timeLeftText(st, now); !left.empty()) l.text += kDot + left;
            return l;
        }
        if (st.state == State::Connecting) {
            l.kind = JobLine::Kind::Busy;
            if (step(st, Step::Login).status == StepStatus::Running) l.text = "Logging in\xE2\x80\xA6";
            else if (step(st, Step::Queue).status == StepStatus::Running) {
                // "job 4711 · PENDING (Resources) · 0:42"
                const std::vector<std::string> parts = dotted(step(st, Step::Queue).detail);
                if (parts.size() >= 3) {
                    std::string state = parts[1], why;
                    if (const std::size_t paren = state.find(" ("); paren != std::string::npos) {
                        why = state.substr(paren + 2);
                        if (!why.empty() && why.back() == ')') why.pop_back();
                        state.resize(paren);
                    }
                    const std::string word = state == "PENDING" ? std::string("queued") : lower(state);
                    l.text = "Job " + st.jobId + " " + word + (why.empty() ? std::string() : " (waiting for " + why + ")") + kDot + "waited " +
                             parts[2];
                } else {
                    l.text = "Job " + st.jobId + " submitted: waiting in the queue\xE2\x80\xA6";
                }
            } else if (step(st, Step::Submit).status == StepStatus::Done)
                l.text = "Job " + st.jobId + " submitted: waiting in the queue\xE2\x80\xA6";
            else
                l.text = "Submitting the job\xE2\x80\xA6";
            return l;
        }
        for (const Step s : {Step::Submit, Step::Queue})
            if (step(st, s).status == StepStatus::Failed) {
                l.kind = JobLine::Kind::Failed;
                l.text = st.reason.empty() ? step(st, s).detail : st.reason;
                return l;
            }
        if (st.state == State::Disconnected && st.jobEnded && !st.reason.empty()) {
            l.kind = JobLine::Kind::Failed;
            l.text = st.reason;
            return l;
        }
        l.text = "No job yet: Start job asks Slurm for one with these settings.";
        return l;
    }

    // --- the worker's health ------------------------------------------------------------------

    HealthReport healthReport(const Status& st, const Profile& p, const BuildInfo& app, std::chrono::steady_clock::time_point now) {
        HealthReport r;
        const auto row = [&r](std::string label, std::string value, Mark mark) { r.rows.push_back(HealthRow{std::move(label), std::move(value), mark}); };
        if (st.state != State::Connected) {
            if (st.state == State::Starting) {
                r.verdict = HealthReport::Verdict::Busy;
                r.headline = "Starting the worker\xE2\x80\xA6";
            } else if (workerStepFailed(st) || (st.state == State::Disconnected && (st.jobEnded || st.dropped) && !st.reason.empty())) {
                r.verdict = HealthReport::Verdict::Failed;
                r.headline = "Not ready: " + st.reason;
                r.fix = st.fix;
                r.details = st.remoteOutput;
            } else {
                r.headline = st.state == State::JobReady && !st.reason.empty() ? "No worker runs: " + st.reason + "."
                                                                               : std::string("Start the worker to see its report.");
            }
            // what the worker's start went through
            for (const Step s : {Step::Checks, Step::Start, Step::Hello}) {
                const StepState& x = step(st, s);
                Mark m = Mark::Info;
                std::string v = x.detail;
                switch (x.status) {
                    case StepStatus::Pending: v = "not yet"; break;
                    case StepStatus::Running: v = x.detail.empty() ? "running\xE2\x80\xA6" : x.detail + "\xE2\x80\xA6"; break;
                    case StepStatus::Done: m = Mark::Ok; break;
                    case StepStatus::Warning: m = Mark::Warn; break;
                    case StepStatus::Failed: m = Mark::Fail; break;
                }
                row(s == Step::Checks ? "Image checks" : (s == Step::Start ? "Worker start" : "Connection"), v, m);
            }
            if (jobHeld(st)) {
                row("Node", st.node + kDot + "job " + st.jobId, Mark::Info);
                if (const std::string left = timeLeftText(st, now); !left.empty()) row("Job time left", left, Mark::Info);
            }
            return r;
        }

        const WorkerCapabilities& c = st.caps;
        const nlohmann::json& engine = c.engine;
        const bool hasEngine = engine.is_object();
        nlohmann::json python = hasEngine && engine.contains("python") ? engine["python"] : nlohmann::json();
        int warnings = 0;
        const auto warn = [&warnings](Mark m) {
            if (m == Mark::Warn || m == Mark::Fail) ++warnings;
            return m;
        };

        row("Node", st.node + kDot + "job " + st.jobId, Mark::Ok);
        // GPUs and CUDA
        if (!c.gpus.empty()) row("GPU", gpuSummary(c.gpus), Mark::Ok);
        else if (p.gpus > 0) row("GPU", "none found (the job asked for " + std::to_string(p.gpus) + ")", warn(Mark::Warn));
        else row("GPU", "none (the job asked for no GPU)", Mark::Info);
        if (gpuUsable(c)) row("CUDA", "usable" + (c.device.empty() ? std::string() : kDot + c.device), Mark::Ok);
        else if (p.gpus <= 0 && c.gpus.empty()) row("CUDA", "not used: the steps run on the CPU", Mark::Info);
        else row("CUDA", "not usable: " + (c.cudaReason.empty() ? std::string("the worker reports no CUDA") : c.cudaReason), warn(Mark::Warn));
        // torch: the Python worker's, or the one beside the engine
        std::string torch = c.torch;
        std::string pythonState = jsonString(python, "state");
        if (torch.empty() && python.is_object() && python.contains("caps")) torch = jsonString(python["caps"], "torch");
        if (!torch.empty()) row("torch", torch, Mark::Ok);
        else if (hasEngine && (pythonState == "starting" || pythonState == "idle" || pythonState.empty()))
            row("torch", "the Python worker beside the engine is still starting", Mark::Info);
        else if (hasEngine && pythonState == "failed")
            row("torch", "the Python worker did not start: " + jsonString(python, "error"), warn(Mark::Warn));
        else row("torch", "not in the image: Python steps that need it fail", warn(Mark::Warn));
        // the sirius package (TIFF on the cluster)
        if (!c.tiffReader.empty()) row("sirius package", c.tiffReader, Mark::Ok);
        else row("sirius package", "not importable in the image: TIFF datasets on the cluster cannot be opened", warn(Mark::Warn));
        bool nvtiff = c.nvtiff;
        if (hasEngine && engine.contains("cuda") && engine["cuda"].is_object() && engine["cuda"].value("nvtiff", false)) nvtiff = nvtiff || gpuUsable(c);
        row("nvTIFF", nvtiff ? std::string("yes: TIFF decoded on the GPU") : std::string("no: TIFF decoded on the CPU"), nvtiff ? Mark::Ok : Mark::Info);
        // the engine
        if (hasEngine) {
            const BuildInfo e = buildInfoFromJson(engine);
            std::string v = e.build.empty() ? e.version : e.build;
            if (!e.commit.empty() && e.commit != "unknown") v += " (" + e.commit.substr(0, std::min<std::size_t>(10, e.commit.size())) + ")";
            const std::string mismatch = engineMismatch(app, e);
            Mark m = Mark::Ok;
            if (!mismatch.empty()) {
                v += kDot + "not compatible with this application";
                m = warn(Mark::Fail);
            } else {
                v += kDot + (e.commit == app.commit ? "this application's build" : "compatible (same operations)");
            }
            v += kDot + (st.engineBuild.empty() ? std::string("the image's own") : "from " + st.engineBuild);
            row("C++ engine", v, m);
        } else {
            row("C++ engine", p.engine ? std::string("not running: only the Python steps run on the node") : std::string("off: only the Python steps run on the node"),
                warn(Mark::Warn));
        }
        // the job's size
        row("CPU threads", (c.cpuThreads > 0 ? std::to_string(c.cpuThreads) + " on the node" : std::string("not said")) + kDot + "the job asked for " + std::to_string(p.cpus),
            Mark::Info);
        row("Memory", p.mem.empty() ? std::string("Slurm's default") : p.mem + " (the job's)", Mark::Info);
        // the data folders
        const std::vector<std::string> binds = bindHostPaths(p.bind);
        std::string folders;
        for (const std::string& b : binds) folders += (folders.empty() ? "" : ", ") + b;
        row("Data folders", folders.empty() ? std::string("your home folder only") : folders + " (and your home folder)", binds.empty() ? Mark::Info : Mark::Ok);
        // the time left
        const std::string left = timeLeftText(st, now);
        Mark lm = Mark::Ok;
        if (st.jobLimitSeconds >= 0 && st.jobStarted != std::chrono::steady_clock::time_point{}) {
            const long long used = std::chrono::duration_cast<std::chrono::seconds>(now - st.jobStarted).count();
            if (st.jobLimitSeconds - used < 600) lm = warn(Mark::Warn);
        }
        row("Job time left", left.empty() ? std::string("not known") : left, left.empty() ? Mark::Info : lm);

        r.verdict = HealthReport::Verdict::Ready;
        r.headline = warnings == 0 ? std::string("Ready") : "Ready, with " + std::to_string(warnings) + (warnings == 1 ? " warning" : " warnings");
        return r;
    }

} // namespace sirius::app::cluster::wizard
