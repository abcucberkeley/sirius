#ifndef SIRIUS_APP_CLUSTER_HPP
#define SIRIUS_APP_CLUSTER_HPP

// A session with a Slurm cluster, from one SSH login to a worker answering:
//
//   Login    ssh to the profile's host (core/remote_host.hpp); the password
//            and one-time-code prompts go to `prompt`
//   Checks   sbatch on the host, the SIRIUS checkout (app/python), the venv
//            and the packages the worker needs in it -- one probe, and what
//            is missing comes back with the command that fixes it
//   Submit   app/python/slurm/sirius_worker.sbatch with the profile's
//            partition, account, QoS, time and resources; the token is made
//            here and written over the command channel to a 0600 file in
//            ~/.sirius/run (a 0700 directory), whose name is all the job is
//            given: never an argument, never the job's environment
//   Queue    squeue every few seconds: state, reason, time waited
//   Start    the job runs on a node; its log (in ~/.sirius/run too) says
//            when the worker listens and on which port (it takes a free one)
//   Hello    the application connects through the SSH session's SOCKS proxy
//            and the worker says what it is (version, device, steps)
//
// Then it keeps watching: ssh alive, the worker answering a ping, the job
// still RUNNING (squeue / sacct say why not: TIMEOUT, CANCELLED, ...). A
// failure at any point leaves the session disconnected with the reason in
// plain words and the remote side's own output; nothing is retried by
// itself. disconnect() closes the connections and cancels the job only when
// asked to.
//
// The SSH session serves the cluster's file system as well (list()), as
// soon as the login is over, before any job runs.
//
// Threads: every public function may be called from any thread; connect()
// works on a thread of its own and reports through `changed` (any thread).
// GUI-free: sirius-cli can drive it.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/remote_host.hpp"
#include "core/rpc.hpp"

namespace sirius::app::cluster {

    struct Profile {
        std::string host = "fiona";                // ssh destination (an alias of ~/.ssh/config works)
        std::string checkout = "~/dev/sirius";     // the SIRIUS checkout on the cluster
        std::string venv = "~/venvs/sirius";       // activated for the worker; "" = none
        std::string partition = "abc_a100";
        std::string account = "velatkilic";
        std::string qos = "abc_debug";
        std::string time = "01:00:00";
        int gpus = 1;
        int cpus = 8;
        std::string mem = "64G";
        int port = 7645;                           // unused: the worker takes a free port (kept for old profiles)
        std::string sshProgram;                    // "" = the system's ssh
        std::vector<std::string> sshProgramArgs;   // tests: a fake ssh run by an interpreter

        nlohmann::json toJson() const;
        static Profile fromJson(const nlohmann::json& j);
    };

    enum class Step { Login,
                      Checks,
                      Submit,
                      Queue,
                      Start,
                      Hello };
    inline constexpr int kStepCount = 6;
    const char* stepTitle(Step s);

    enum class StepStatus { Pending,
                            Running,
                            Done,
                            Failed,
                            Warning };

    enum class State { Idle,           // never connected this session
                       Connecting,
                       Connected,
                       Disconnected };   // after a failure or disconnect(); `reason` says which

    struct StepState {
        StepStatus status = StepStatus::Pending;
        std::string detail;    // one line: "job 4711 · PENDING (Resources) · 0:42"
    };

    struct Status {
        State state = State::Idle;
        bool sshUp = false;
        std::vector<StepState> steps = std::vector<StepState>(kStepCount);
        std::string reason;          // why it is disconnected or what failed, in plain words
        std::string remoteOutput;    // the remote side's own words for it (stderr, the job log)
        std::string fix;             // a command that fixes what the checks found missing
        std::string jobId, node, jobState;
        WorkerCapabilities caps;
        std::string host;
        std::chrono::steady_clock::time_point since{};   // when `state` began
    };

    struct Entry {
        std::string name;
        bool dir = false;
        bool link = false;
        std::uint64_t size = 0;
        double mtime = 0.0;     // seconds since the epoch
    };
    struct Listing {
        std::string path;       // absolute, as the cluster resolved it
        std::string home;
        std::vector<Entry> entries;
        bool truncated = false;
    };
    // The listing script's JSON line (exposed for tests).
    Listing parseListing(const std::string& line);
    // The bash script that lists `path` (exposed for tests).
    std::string listingScript(const std::string& path, int maxEntries);

    class Session {
    public:
        // Called on the session's thread when ssh asks; blocks until the user
        // answers. nullopt cancels the login: ssh is stopped before the
        // helper hears back, so nothing is sent to the server.
        using PromptFn = std::function<std::optional<std::string>(const ssh::Prompt&)>;

        Session();
        ~Session();   // stops the threads and ssh; leaves the job as it is
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;

        void setPrompt(PromptFn fn);
        // The askpass helper: this application's executable (ssh::askpassMain).
        void setAskpassProgram(std::string program);
        // Any thread: something in status() changed.
        void setChanged(std::function<void()> fn);
        // Any thread: a line for the application's log (each state change).
        void setLog(std::function<void(const std::string&)> fn);
        // Test hooks: shorter polls.
        void setPollInterval(std::chrono::milliseconds queue, std::chrono::milliseconds keepAlive);

        // Starts the steps on the session's thread (from Login, or from
        // Submit when the SSH login of this profile's host is still up).
        // Ignored while connecting.
        void connect(const Profile& profile);
        // Stops a connect in progress (the prompt included).
        void cancelConnect();
        // Closes the worker connection and ssh; with `cancelJob` scancels the
        // job first. Blocks for as long as scancel takes (seconds).
        void disconnect(bool cancelJob);

        Status status() const;
        Profile profile() const;
        bool connected() const;
        bool sshUp() const;
        bool hasJob() const;      // a job this session submitted that may still run

        // Where a worker connection goes: the node, the port, the token and
        // the proxy. Empty host while not connected.
        struct Endpoint {
            std::string host;
            int port = 0;
            std::string token;
            int socksPort = 0;
        };
        Endpoint endpoint() const;
        // A new connection to the worker (another client slot of it).
        std::unique_ptr<RemoteWorker> connectWorker(std::chrono::milliseconds timeout = std::chrono::seconds(10),
                                                    const std::function<bool()>& cancelled = {}) const;

        // The cluster's file system through the SSH session; throws
        // ssh::SshError (not logged in, a path that cannot be read).
        Listing list(const std::string& path, int maxEntries = 5000);
        ssh::CommandResult run(const std::string& script, std::chrono::milliseconds timeout = std::chrono::seconds(60));

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::cluster

#endif // SIRIUS_APP_CLUSTER_HPP
