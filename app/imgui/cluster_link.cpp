#include "imgui/cluster_link.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include <nlohmann/json.hpp>

#include "core/host.hpp"
#include "core/errors.hpp"
#include "core/secure_wipe.hpp"
#include "imgui/app.hpp"
#include "imgui/settings.hpp"
#include "imgui/theme.hpp"

namespace sirius::app::gui {

    namespace {
        constexpr int kMaxRecentFolders = 10;

        // ssh runs this executable as its askpass helper (main() hands such a
        // start to ssh::askpassMain before anything else).
        std::string askpassProgram() {
#ifdef _WIN32
            return host::executableDirectory() + "/sirius-app.exe";
#else
            return host::executableDirectory() + "/sirius-app";
#endif
        }

        constexpr ImU32 kConnected = theme::rgb(0x2e, 0x7d, 0x32);
    } // namespace

    ClusterLink::ClusterLink(App& app) : app_(app), alive_(std::make_shared<std::atomic<bool>>(true)) {
        session_.setAskpassProgram(askpassProgram());
        session_.setPrompt([this](const ssh::Prompt& p) { return ask(p); });
        Bridge& bridge = app.bridge();
        auto alive = alive_;
        session_.setLog([alive, &bridge, this](const std::string& line) {
            bridge.post([alive, this, line] {
                if (alive->load()) app_.wb().logLine(line);
            });
        });
        // a status change wakes the frame loop, whose frame() reads it
        session_.setChanged([alive, &bridge] { bridge.post([] {}); });
    }

    ClusterLink::~ClusterLink() {
        alive_->store(false);
        session_.cancelConnect();
        if (disconnecting_.joinable()) disconnecting_.join();
        if (datasets_) datasets_->uninstall();
    }

    cluster::Profile ClusterLink::storedProfile() const {
        Settings& s = settings();
        if (!s.contains("cluster/profile")) return cluster::Profile{};
        return cluster::Profile::fromJson(s.value("cluster/profile"));
    }

    void ClusterLink::connect(const cluster::Profile& profile) {
        lastState_ = cluster::State::Connecting;
        session_.connect(prepared(profile));
    }

    void ClusterLink::logIn(const cluster::Profile& profile) {
        lastState_ = cluster::State::Connecting;
        session_.logIn(prepared(profile));
    }

    cluster::Profile ClusterLink::prepared(const cluster::Profile& profile) {
        // saved, with its Slurm choice remembered for its host
        cluster::Profile saved = profile;
        saved.remember();
        settings().set("cluster/profile", saved.toJson());
        cluster::Profile p = saved;
        // Tests and screenshots only: $SIRIUS_TEST_SSH, a JSON list (program and
        // its first arguments), stands in for ssh -- tests/tools/fake_ssh.py.
        if (const std::string fake = host::environment("SIRIUS_TEST_SSH"); !fake.empty()) {
            // never the real ssh in its place: a value that does not parse connects nothing
            p.sshProgram = "<SIRIUS_TEST_SSH is not a JSON list>";
            try {
                const nlohmann::json j = nlohmann::json::parse(fake);
                if (j.is_array() && !j.empty() && j[0].is_string()) {
                    p.sshProgram = j[0].get<std::string>();
                    p.sshProgramArgs.clear();
                    for (std::size_t i = 1; i < j.size(); ++i) p.sshProgramArgs.push_back(j[i].get<std::string>());
                }
            } catch (const std::exception&) {
            }
        }
        return p;
    }

    void ClusterLink::disconnect(bool cancelJob) {
        if (disconnecting_.joinable()) disconnecting_.join();
        disconnecting_ = std::thread([this, cancelJob] { session_.disconnect(cancelJob); });
    }

    void ClusterLink::disconnectAsking() {
        const cluster::Status st = status();
        if (st.jobId.empty()) {
            disconnect(false);
            return;
        }
        app_.ask("Disconnect from " + st.host, "Cancel the worker job " + st.jobId + " on " + st.host + " as well? Left running, it keeps its GPU until its time limit, and Connect starts a new one.", {"Leave it running", "Cancel the job"}, [this](int answer) {
                     if (answer < 0) return;
                     disconnect(answer == 1); }, 1);
    }

    RemoteConfig ClusterLink::remoteConfig() const {
        RemoteConfig rc;
        const cluster::Session::Endpoint e = session_.endpoint();
        rc.host = e.host;
        rc.port = e.port;
        rc.token = e.token;
        rc.socksPort = e.socksPort;
        const cluster::Status st = session_.status();
        if (st.state == cluster::State::Connected) {
            // every connection the session's endpoint as it is then: a
            // reconnect to the same job keeps the results' handles working
            const cluster::Session* session = &session_;
            auto alive = alive_;
            rc.connect = [alive, session](const std::function<bool()>& cancelled) {
                if (!alive->load()) throw ProtocolError("the application is closing");
                return session->connectWorker(std::chrono::seconds(10), cancelled);
            };
            rc.known = true;
            rc.engine = st.caps.engine;
            rc.where = st.host + " \xC2\xB7 " + st.node + (st.jobId.empty() ? std::string() : " \xC2\xB7 job " + st.jobId);
        }
        return rc;
    }

    std::optional<std::string> ClusterLink::ask(const ssh::Prompt& p) {
        // on the askpass relay's thread
        // nobody to answer in a scripted run (but a screenshot of the box against the fake ssh)
        if (app_.unattended() && host::environment("SIRIUS_TEST_SSH").empty()) return std::nullopt;
        auto req = std::make_shared<ClusterPrompt>();
        req->prompt = p;
        req->host = session_.profile().host;
        auto alive = alive_;
        app_.bridge().post([alive, this, req] {
            if (!alive->load()) {
                req->answered.store(true);
                return;
            }
            app_.showDialog(makeClusterPromptDialog(app_, req));
        });
        while (!req->answered.load()) {
            if (!alive_->load() || session_.status().state != cluster::State::Connecting) {
                req->abandoned.store(true);
                return std::nullopt;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (req->cancelled) {
            secureWipe(req->answer);
            return std::nullopt;
        }
        // Copied, then the source wiped: a move leaves a short string's bytes
        // where they were (the small-string buffer), and clear() keeps them.
        std::optional<std::string> answer = req->answer;
        secureWipe(req->answer);
        return answer;
    }

    void ClusterLink::frame() {
        // the cluster's datasets are decoded where the session computes
        // (nvTIFF on the job's GPU), switched with the HPC device
        if (datasets_) datasets_->setDevice(app_.wb().hpcDevice() == HpcDevice::Cpu ? "cpu" : "cuda");
        const cluster::State now = session_.status().state;
        if (now == lastState_) return;
        const cluster::State before = lastState_;
        lastState_ = now;
        if (now == cluster::State::Connected) {
            onState(session_.status());
        } else if (before == cluster::State::Connected) {
            if (datasets_) datasets_->uninstall();
            datasets_.reset();
            // The job ended: the engine's results went with it. A job left
            // running keeps them for a reconnect (which reattaches to it).
            const cluster::Status st = session_.status();
            if (st.jobEnded && !app_.wb().engineSession().empty()) {
                const int n = app_.wb().nodeOutputsGone(app_.wb().engineSession(), "held by the cluster job, which ended (" + st.reason + ")");
                if (n > 0) app_.wb().logLine("HPC: the results of " + std::to_string(n) + " step(s) went with the job: run them again to see them.");
            }
        }
        app_.requestRedraw();
    }

    void ClusterLink::onState(const cluster::Status& st) {
        auto alive = alive_;
        datasets_ = std::make_shared<RemoteDatasets>(st.host, [alive, this]() -> std::unique_ptr<RemoteWorker> {
            if (!alive->load()) throw ProtocolError("the application is closing");
            return session_.connectWorker();
        });
        datasets_->install();
        Workbench& wb = app_.wb();
        // a job without a GPU computes on its CPU; with one, the session's choice stands
        if (!cluster::gpuUsable(st.caps)) wb.setHpcDevice(HpcDevice::Cpu);
        datasets_->setDevice(wb.hpcDevice() == HpcDevice::Cpu ? "cpu" : "cuda");
        // Another engine than the one the results here came from (a new job):
        // those are gone; the same one (a reattached job) still holds them.
        const std::string session = st.caps.engine.is_object() ? st.caps.engine.value("session", std::string()) : std::string();
        if (!wb.engineSession().empty() && wb.engineSession() != session)
            wb.nodeOutputsGone(wb.engineSession(), "held by an earlier cluster job, which has ended");
        wb.setRemoteConfig(remoteConfig());
        wb.setBackend(Backend::Hpc);
        if (st.caps.engine.is_object())
            wb.logLine("HPC: SIRIUS's engine on " + st.node + " (" + st.caps.device + ", job " + st.jobId +
                       ") runs every step there; its results stay there until shown or exported. Cluster datasets open from " + st.host);
        else
            wb.logLine("HPC: the Python worker on " + st.node + " (" + st.caps.device + ") runs the Python steps (no SIRIUS engine in this job: "
                                                                                        "built-in steps are refused on HPC); cluster datasets open from " +
                       st.host);
    }

    std::string ClusterLink::indicator(ImU32& color) const {
        const cluster::Status st = status();
        switch (st.state) {
            case cluster::State::Idle: return {};
            case cluster::State::Connecting: {
                color = theme::kNeutral700;
                std::string step;
                for (int i = 0; i < cluster::kStepCount; ++i)
                    if (st.steps[static_cast<std::size_t>(i)].status == cluster::StepStatus::Running)
                        step = cluster::stepTitle(static_cast<cluster::Step>(i));
                return "HPC: connecting\xE2\x80\xA6" + (step.empty() ? std::string() : " (" + step + ")");
            }
            case cluster::State::Connected:
                color = kConnected;
                return "HPC: " + st.node + " \xC2\xB7 " + toString(app_.wb().hpcDevice());
            case cluster::State::Disconnected: color = theme::kAccentText; return "HPC: disconnected: " + st.reason;
        }
        return {};
    }

    bool ClusterLink::hpcGpuUsable(std::string* why) const {
        const cluster::Status st = status();
        if (st.state == cluster::State::Connected) {
            if (cluster::gpuUsable(st.caps)) return true;
            if (why) *why = cluster::gpuUnusableReason(st.node, st.caps);
            return false;
        }
        const cluster::Profile p = st.state == cluster::State::Idle ? storedProfile() : session_.profile();
        if (p.gpus > 0) return true;
        if (why) *why = "The cluster profile asks for no GPU (GPUs 0): set GPUs \xE2\x89\xA5 1 in Connect to cluster to use one";
        return false;
    }

    std::vector<cluster::NodeDevice> ClusterLink::nodeDevices() const {
        const cluster::Status st = status();
        if (st.state != cluster::State::Connected) return {};
        return cluster::nodeDevices(st.node, st.caps);
    }

    std::vector<std::string> ClusterLink::recentFolders() const { return settings().getStringList("cluster/recentFolders"); }

    void ClusterLink::addRecentFolder(const std::string& path) {
        std::vector<std::string> list = recentFolders();
        list.erase(std::remove(list.begin(), list.end(), path), list.end());
        list.insert(list.begin(), path);
        if (list.size() > static_cast<std::size_t>(kMaxRecentFolders)) list.resize(static_cast<std::size_t>(kMaxRecentFolders));
        settings().set("cluster/recentFolders", list);
    }

    void ClusterLink::settleBeforeQuit(std::function<void()> done) {
        const cluster::Status st = status();
        if (st.jobId.empty() || !st.sshUp) {
            session_.cancelConnect();
            done();
            return;
        }
        app_.ask("Quit", "The worker job " + st.jobId + " on " + st.host + " is still running: it keeps its GPU until its time limit. Cancel it before quitting?", {"Leave it running", "Cancel the job"}, [this, done](int answer) {
                     if (answer < 0) return;   // the quit is called off
                     if (disconnecting_.joinable()) disconnecting_.join();
                     session_.disconnect(answer == 1);   // scancel: seconds at most
                     done(); }, 1);
    }

} // namespace sirius::app::gui
