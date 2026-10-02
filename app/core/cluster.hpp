#ifndef SIRIUS_APP_CLUSTER_HPP
#define SIRIUS_APP_CLUSTER_HPP

// A session with a Slurm cluster, from one SSH login to a worker answering:
//
//   Login    ssh to the profile's host (core/remote_host.hpp); the password
//            and one-time-code prompts go to `prompt`; then the cluster's
//            partitions and the user's associations (ClusterInfo), asked
//            once per host (logIn() does only this much, and asks again)
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
#include <map>
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

    // The Slurm part of a profile as last chosen on one host.
    struct SlurmChoice {
        std::string partition, account, qos, time;
    };

    struct Profile {
        std::string host = "fiona";                // ssh destination (an alias of ~/.ssh/config works)
        std::string checkout = "~/dev/sirius";     // the SIRIUS checkout on the cluster
        std::string venv = "~/venvs/sirius";       // activated for the worker; "" = none; unused with a container
        // An Apptainer/Singularity image the worker runs in (the compiled
        // sirius package, numpy, torch): `<launcher> exec [--nv] --bind
        // <checkout> --bind ~/.sirius/run <container> python -m sirius_worker`.
        // "" = the venv, as before.
        std::string container;
        std::string launcher = "apptainer";        // singularity is tried when it is not found
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
        std::map<std::string, SlurmChoice> perHost;   // partition, account, QoS and time last used on each host

        // perHost[host] = this profile's partition, account, QoS and time.
        void remember();
        // Takes the partition, account, QoS and time last used on `host`, if
        // any were; false when there were none (the fields stay).
        bool recall(const std::string& host);

        nlohmann::json toJson() const;
        static Profile fromJson(const nlohmann::json& j);
    };

    // --- what the cluster offers -----------------------------------------------------
    //
    // Asked over the command channel once the login is over (and again on
    // Refresh); every command is fixed text, the user's name is $USER on the
    // cluster:
    //
    //   sinfo -h -o '%P|%a|%l|%D|%t|%G|%c|%m'                 the partitions and their nodes
    //   sacctmgr -n -P show assoc user="$u" format=partition,account,qos,defaultqos
    //                                                          where the user may submit
    //   sacctmgr -n -P show qos format=name,maxwall            each QoS's time limit
    //   scontrol -o show partition                             whole-node partitions
    //
    // Only sinfo has to answer; the others may be refused (a site that hides
    // its accounting) and their part is then unknown, not empty.

    struct Partition {
        std::string name;
        bool isDefault = false;         // sinfo's '*'
        bool up = true;                 // sinfo %a
        std::string maxTime;            // as Slurm says it: "3-00:00:00", "infinite"; "" unknown
        int nodes = 0, idle = 0, mixed = 0;
        int gpusPerNode = 0;            // the most of any of its nodes (gres gpu)
        std::string gpuType;            // "a100"; "" when the gres names none
        int cpusPerNode = 0;            // the most of any of its nodes
        long long memPerNodeMB = 0;     // the most of any of its nodes
        bool exclusive = false;         // scontrol: OverSubscribe=EXCLUSIVE, a job takes whole nodes
    };

    struct Association {
        std::string partition;          // "" = every partition
        std::string account;
        std::vector<std::string> qos;
        std::string defaultQos;
    };

    struct ClusterInfo {
        std::string host, user;
        std::vector<Partition> partitions;            // in sinfo's order
        std::vector<Association> associations;
        bool associationsKnown = false;               // sacctmgr answered
        std::map<std::string, std::string> qosMaxWall;   // QoS -> its MaxWall ("" none)
        bool exclusiveKnown = false;                  // scontrol answered
        std::string error;                            // sinfo did not answer: why
        std::vector<std::string> notes;               // what else could not be asked
    };

    // The script that asks (above) and the reading of its output (exposed for tests).
    std::string clusterInfoScript();
    ClusterInfo parseClusterInfo(const std::string& output);

    const Partition* findPartition(const ClusterInfo& info, const std::string& name);
    // Whether the user has an association for `partition` (true when sacctmgr did not say).
    bool hasAssociation(const ClusterInfo& info, const std::string& partition);
    // The accounts the user may submit to `partition` with, the partition's
    // own associations first; and the QoS of one of them.
    std::vector<std::string> accountsFor(const ClusterInfo& info, const std::string& partition);
    std::vector<std::string> qosFor(const ClusterInfo& info, const std::string& partition, const std::string& account);
    // "4 nodes (2 idle) · 1x A100 per node · max 3-00:00:00"; `full` adds CPUs and memory.
    std::string partitionSummary(const Partition& p, bool full = false);
    // What to know before submitting there (whole nodes, GPUs shared by a group); "" nothing.
    std::string partitionWarning(const Partition& p);

    // Slurm's time formats ("90", "1:30:00", "2-12", "3-00:00:00") in
    // seconds; -1 unlimited ("infinite", "UNLIMITED"), -2 unreadable.
    long long slurmTimeSeconds(const std::string& text);
    std::string slurmTimeText(long long seconds);   // "01:30:00", "3-00:00:00"
    // "64G", "500M", "1T", "4096" (MB, Slurm's default unit) in MB; -1 unreadable.
    long long memoryMB(const std::string& text);

    // Choosing `partition` from the list: the account and QoS from the
    // user's association with it (the profile's own kept when they are
    // among them), and the time, GPUs, CPUs and memory brought within what
    // its nodes and the QoS allow. Returns what changed, in words.
    std::vector<std::string> choosePartition(Profile& p, const ClusterInfo& info, const std::string& partition);

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
        // The SSH login only (kept when it is up for this host), then the
        // cluster's partitions (clusterInfo()): no job is submitted. The
        // session is Idle again afterwards, logged in.
        void logIn(const Profile& profile);
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

        // What the cluster offers, as last asked (on a login, or refreshed);
        // kept after a disconnect. nullopt before the first answer.
        std::optional<ClusterInfo> clusterInfo() const;
        // Asks again (blocks for as long as the commands take); throws
        // ssh::SshError when not logged in. Any thread.
        ClusterInfo refreshClusterInfo();
        bool queryingClusterInfo() const;

    private:
        void start(const Profile& profile, bool submit);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::cluster

#endif // SIRIUS_APP_CLUSTER_HPP
