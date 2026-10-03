#ifndef SIRIUS_APP_CLUSTER_HPP
#define SIRIUS_APP_CLUSTER_HPP

// A session with a Slurm cluster, in two steps.
//
// The job (connectJob):
//   Login    ssh to the profile's host (core/remote_host.hpp); the password
//            and one-time-code prompts go to `prompt`; then the cluster's
//            partitions and the user's associations (ClusterInfo), asked
//            once per host (logIn() does only this much, and asks again)
//   Submit   a job that only holds the allocation (sbatch --wrap: it sleeps
//            until its time limit or until it is cancelled), with the
//            profile's partition, account, QoS, time and resources; its log
//            is in ~/.sirius/run (a 0700 directory). Nothing of SIRIUS has to
//            be on the cluster for this.
//   Queue    squeue every few seconds: state, reason, time waited, until the
//            job runs on a node.
//
// The worker (startWorker), inside that job:
//   Checks   the worker's code (the engine build's python/ folder, else the
//            checkout's app/python), the container image (there, readable,
//            sirius and numpy import in it), the launcher, the data folders,
//            SIRIUS's engine when the profile runs it -- the engine build
//            that fits this application (Profile::engineBuilds), the named
//            executable, or the image's own, each made sure of: never a
//            silent fall back to the Python worker alone -- and the node
//            cache folder: what is missing comes back with what to do about it
//   Start    the application's own launch script (sirius_worker.sbatch as
//            compiled into it, workerLaunchScript) is written over the
//            command channel to ~/.sirius/run/<workerLaunchScriptName()>
//            (0700), and srun --jobid=<job> --overlap runs it as a step of
//            the job: SIRIUS's engine (or the Python worker) in the image. The
//            checkout's copy of the script is never run: it may be older than
//            the application. The token is made here and written over the
//            command channel to a 0600 file in ~/.sirius/run whose name is all
//            the step is given: never an argument, never the job's
//            environment. The step's log says when the worker listens and on
//            which port (it takes a free one).
//   Hello    the application connects through the SSH session's SOCKS proxy
//            and the worker says what it is (version, device, steps): a
//            protocol or version other than this application's, SIRIUS's
//            engine whose operations are not this application's
//            (core/build_info.hpp), and a worker without the engine the
//            profile asked for are refused here, each with its fix
//
// A new image, new data folders or another engine only restart the worker
// step in the same job (startWorker again); another partition, account, QoS,
// time or size needs a new job. connect() does both steps in one go.
//
// Connect after a disconnect that left the job running reattaches to that
// job when it still runs (no new submission), and to its worker step when
// that still answers: its engine still holds what it computed.
//
// Then it keeps watching: ssh alive, the worker answering a ping, the job
// still RUNNING (squeue / sacct say why not: TIMEOUT, CANCELLED, ...). A
// worker that stops leaves the job held (JobReady); a job that ends or an SSH
// connection that drops leaves the session disconnected with the reason in
// plain words and the remote side's own output; nothing is retried by
// itself. disconnect() closes the connections and cancels the job only when
// asked to.
//
// buildImage() builds a worker image in the held job (apptainer build
// --fakeroot from a definition file), when the cluster allows it.
//
// The SSH session serves the cluster's file system as well (list()), as
// soon as the login is over, before any job runs.
//
// Threads: every public function may be called from any thread; the steps
// run on a thread of their own and report through `changed` (any thread).
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

#include "core/build_info.hpp"
#include "core/remote_host.hpp"
#include "core/rpc.hpp"

namespace sirius::app::cluster {

    // The Slurm part of a profile as last chosen on one host, and the
    // container's binds (the cluster's own paths). The binds are optional: a
    // choice saved before they were kept leaves the profile's as they are.
    struct SlurmChoice {
        std::string partition, account, qos, time;
        std::optional<std::string> bind, containerPythonPath;
    };

    // One partition a profile's dropdowns offer, with what goes with it:
    // the accounts and QoS to submit there with, the longest time, and the
    // resources a job there gets by default and at most. Written in the
    // settings file (core/cluster_profiles.hpp), by hand or added from what
    // the cluster reports (choiceFromCluster).
    struct PartitionChoice {
        std::string name;
        bool isDefault = false;              // the one a new job uses
        std::vector<std::string> accounts;   // first = the one picked with it
        std::vector<std::string> qos;        // first = the one picked with it
        std::string maxTime;                 // Slurm's form, "3-00:00:00"; "" = not said
        std::vector<std::string> times;      // time limits offered; empty = the usual ones up to maxTime
        int gpus = -1, cpus = -1;            // what picking it sets; -1 = leave as is
        std::string mem;                     // "" = leave as is
        int maxGpus = -1, maxCpus = -1;      // a node's; -1 = not said
        std::string maxMem;                  // "" = not said
    };

    // How to run SIRIUS's worker on one cluster. Nothing in it is any
    // site's: a new profile is empty but for the job's size; the login
    // fills the rest from the cluster itself (fillFromCluster).
    //
    // The worker always runs in a container image (Apptainer/Singularity):
    // a cluster needs nothing installed but the launcher, the image and the
    // SIRIUS checkout the worker's code comes from.
    struct Profile {
        // Its name in the list of profiles ("" = its host).
        std::string name;
        std::string host;                          // ssh destination: an alias of ~/.ssh/config, or user@host
        std::string checkout;                      // the SIRIUS checkout on the cluster; "" = <home>/sirius after the login
        // The image the worker runs in (the compiled sirius package, numpy,
        // torch): `<launcher> exec [--nv] --bind <checkout> --bind
        // ~/.sirius/run <container> ...`. Required: Connect stops without one.
        std::string container;
        std::string launcher = "apptainer";        // singularity is tried when it is not found
        // The data folders the image may see besides itself and $HOME,
        // apptainer's --bind syntax: "/data:/data,/scratch/me"
        // (src[:dst[:opts]], comma separated). The job gets it as
        // SIRIUS_CONTAINER_BIND. "" = nothing more.
        std::string bind;
        // Entries appended to the worker's PYTHONPATH inside the image
        // (SIRIUS_CONTAINER_PYTHONPATH); "" = none.
        std::string containerPythonPath;
        std::string partition;                     // "" = the cluster's default partition
        std::string account;                       // "" = the account of your association with it
        std::string qos;                           // "" = the association's default QoS
        std::string time = "01:00:00";
        int gpus = 1;
        int cpus = 8;
        std::string mem = "64G";
        int port = 7645;                           // unused: the worker takes a free port (kept for old profiles)
        // The job runs SIRIUS's engine (`sirius-cli serve`, core/engine_server.hpp)
        // with the Python worker as its child, so every step runs on the node
        // (SIRIUS_ENGINE=1 for sirius_worker.sbatch). Off runs the Python
        // worker alone, which runs only the Python steps.
        bool engine = true;
        // Worker images used with this profile before, newest first: the
        // Worker image dropdown. Data folders likewise (each a --bind list).
        std::vector<std::string> images, bindSets;
        // The definition file a new image is built from (buildImage); "" =
        // <checkout>/containers/sirius-worker.def.
        std::string defFile;
        // A folder of engine builds, one per SIRIUS commit:
        // <engineBuilds>/<commit>/bin/sirius-cli with <commit>/BUILD.json.
        // The checks pick the one of this application's build, or one with
        // the same operations (engineMismatch), and bind it into the image.
        // "" = the image's own engine (/opt/sirius/bin/sirius-cli).
        std::string engineBuilds;
        // The engine's executable, named outright (it wins over engineBuilds); "" = as above.
        std::string engineBin;
        // The node cache folder: where the engine keeps the steps' results
        // and the uploads on the node (sirius-cli serve --scratch, through
        // SIRIUS_ENGINE_SCRATCH; bound into the image). "" = the node's
        // temporary folder. The user's to choose: the checks test it.
        std::string scratch;
        // What the dropdowns offer (clusters.toml); empty until written or
        // added from what the cluster reports.
        std::vector<PartitionChoice> choices;
        std::string sshProgram;                    // "" = the system's ssh
        std::vector<std::string> sshProgramArgs;   // tests: a fake ssh run by an interpreter
        std::map<std::string, SlurmChoice> perHost;   // partition, account, QoS, time and binds last used on each host

        // perHost[host] = this profile's partition, account, QoS, time and binds.
        void remember();
        // Takes the partition, account, QoS, time and binds last used on
        // `host`, if any were; false when there were none (the fields stay).
        bool recall(const std::string& host);
        // The name shown: `name`, else the host, else "New cluster".
        std::string displayName() const;
        const PartitionChoice* choice(const std::string& partition) const;

        // As the settings file keeps it ([cluster.<name>], core/cluster_profiles.hpp).
        // fromJson reads the flat profile of before ("cluster/profile":
        // container, bind, partition, ...) too; keys of the Python
        // environment mode of before ("venv") are ignored.
        nlohmann::json toJson() const;
        static Profile fromJson(const nlohmann::json& j);
    };

    // What changed between the profile a session runs and an edited one:
    // a new job (host, partition, account, QoS, time, GPUs, CPUs, memory),
    // or only a new worker in the same job (checkout, image, launcher, data
    // folders, Python path, engine, node cache folder). Names the fields.
    struct ProfileChange {
        bool newJob = false;
        bool newWorker = false;
        std::vector<std::string> fields;
    };
    ProfileChange profileChange(const Profile& running, const Profile& edited);

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

    // --- the container's binds --------------------------------------------------------

    // The host side of each entry of a --bind list ("/data:/data,
    // /global/scratch:/scratch:ro" -> "/data", "/global/scratch"), in
    // order, blanks and empty entries left out.
    std::vector<std::string> bindHostPaths(const std::string& bind);
    // The warning for a container profile without binds; "" when it has
    // some, or runs no container.
    std::string emptyBindWarning(const Profile& p);
    // Whether the worker in `p`'s container can see `remotePath` (absolute,
    // on the cluster): inside $HOME (`home`; a "~" in the profile stands for
    // it), the checkout, /tmp or a bind's host path. "" when it can (or there
    // is no container, or `home` is unknown); otherwise what to tell the user.
    std::string unboundPathMessage(const Profile& p, const std::string& home, const std::string& remotePath);

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
        std::string home;                             // the user's $HOME there ("" unknown)
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

    // What a profile leaves empty, from what the cluster said at the login:
    // the partition (sinfo's default, '*'), the account and QoS (the
    // user's association with it), and the checkout (<home>/sirius). Fields
    // already set stay. Returns what was filled, in words.
    std::vector<std::string> fillFromCluster(Profile& p, const ClusterInfo& info);

    // --- the node's devices, as the application names them -------------------------
    // A GPU's name without its vendor and form factor: "NVIDIA A100-SXM4-80GB"
    // -> "A100", "NVIDIA GeForce RTX 4090" -> "GeForce RTX 4090".
    std::string shortGpuName(const std::string& name);
    // "1× A100 80 GB", "2× A100 80 GB + 1× V100 32 GB"; "" for none.
    std::string gpuSummary(const std::vector<GpuInfo>& gpus);
    // "g0003.abc0" -> "g0003"; an address ("10.0.0.5") stays whole.
    std::string shortNodeName(const std::string& node);
    // Whether the session's GPU can compute on `caps`'s worker (its CUDA,
    // not only its hardware).
    bool gpuUsable(const WorkerCapabilities& caps);
    // Whether `caps`'s worker is SIRIUS's C++ engine (its hello's "engine"
    // block): what every run on the HPC backend needs.
    bool hasEngine(const WorkerCapabilities& caps);
    // Why the GPU of the worker on `node` cannot be chosen, a sentence; ""
    // when it can. A GPU the node has but the worker cannot use is named,
    // with the worker's reason ("no CUDA library in the worker's environment: ...").
    std::string gpuUnusableReason(const std::string& node, const WorkerCapabilities& caps);
    // The hardware the worker cannot use, for the log and the Hello step:
    // "1× A100 80 GB not usable: no CUDA library ..."; "" when there is none,
    // or it is usable.
    std::string unusableGpuNote(const WorkerCapabilities& caps);

    // What a connected node computes on, one entry each for its GPU and its
    // CPU (the HPC device: Workbench::hpcDevice), in that order: the title
    // bar's device list while the backend is HPC.
    struct NodeDevice {
        bool gpu = false;
        std::string label;     // "g0003 · 1× A100 80 GB", "g0003 · CPU · 16 threads"
        bool usable = true;
        std::string why;       // !usable: the worker's reason, short ("no CUDA library in ...")
    };
    std::vector<NodeDevice> nodeDevices(const std::string& node, const WorkerCapabilities& caps);

    // The steps of a session, in order: the job's three, then the worker's.
    enum class Step { Login,
                      Submit,
                      Queue,
                      Checks,
                      Start,
                      Hello };
    inline constexpr int kStepCount = 6;
    inline constexpr int kJobStepCount = 3;   // Login, Submit, Queue: the job
    const char* stepTitle(Step s);

    enum class StepStatus { Pending,
                            Running,
                            Done,
                            Failed,
                            Warning };

    enum class State { Idle,           // never connected this session (or logged in only)
                       Connecting,     // logging in, getting the job
                       JobReady,       // the job runs on its node; no worker (yet, or any more)
                       Starting,       // the worker starting in the job
                       Connected,      // the worker answers
                       Disconnected };   // after a failure or disconnect(); `reason` says which

    struct StepState {
        StepStatus status = StepStatus::Pending;
        std::string detail;    // one line: "job 4711 · PENDING (Resources) · 0:42"
    };

    // A worker image built in the job (Session::buildImage).
    struct BuildStatus {
        enum class Phase { None,
                           Probing,     // can this cluster build images (fakeroot)?
                           Building,
                           Done,
                           Failed };
        Phase phase = Phase::None;
        // The probe's answer: nullopt not asked, true the cluster builds images,
        // false it does not (`why` says what it said).
        std::optional<bool> supported;
        std::string why;
        std::string image;         // the image being (or that was) built
        std::string log;           // the build's last lines
        std::string error;         // Failed: what went wrong, in words
        std::chrono::steady_clock::time_point started{};
    };

    struct Status {
        State state = State::Idle;
        bool sshUp = false;
        std::vector<StepState> steps = std::vector<StepState>(kStepCount);
        std::string reason;          // why it is disconnected or what failed, in plain words
        std::string remoteOutput;    // the remote side's own words for it (stderr, the job log)
        std::string fix;             // what fixes what the checks found missing
        std::string home;            // the cluster's $HOME, once known ("" unknown)
        std::string jobId, node, jobState;
        // The engine build the checks picked ("" the image's own), and in words.
        std::string engineBuild, engineBuildNote;
        // The worker step did not start, or was refused at its hello, because
        // SIRIUS's engine is not there (none found, or a worker without it):
        // `reason` and `fix` say which and what to do.
        bool noEngine = false;
        // Disconnected because the job ended (it was cancelled, timed out,
        // failed): what its engine held went with it.
        bool jobEnded = false;
        // Disconnected without being asked to: the SSH connection or the job
        // went away while it was held.
        bool dropped = false;
        WorkerCapabilities caps;
        std::string host;
        std::chrono::steady_clock::time_point since{};   // when `state` began
        // The job's time limit in seconds (-1 unlimited, -2 unknown, as
        // slurmTimeSeconds) and when it began to run (squeue's elapsed time
        // taken off): the time left.
        long long jobLimitSeconds = -2;
        std::chrono::steady_clock::time_point jobStarted{};
        BuildStatus build;
    };

    // "42 min", "1 h 05 min", "2 d 3 h"; "" for a negative count.
    std::string durationText(long long seconds);

    // The title bar's cluster button: its label, colour kind and tooltip.
    struct ConnectionBadge {
        enum class Kind { Off,          // no session (or logged in only): "Cluster"
                          Connecting,   // "Connecting…" / "Starting worker…" with `progress`
                          JobReady,     // "g0003 · job 4711 · no worker yet"
                          Connected,    // "g0003 · GPU"
                          Failed,       // a connect that did not get through: "Cluster: failed"
                          Lost };       // the session dropped: "Cluster: lost"
        Kind kind = Kind::Off;
        std::string label;
        std::string tooltip;            // host, job, node, device, time left; or why
        float progress = 0.0f;          // Connecting: the steps done, 0..1
    };
    // `gpu`: the session computes on the node's GPU (else its CPU).
    ConnectionBadge connectionBadge(const Status& st, bool gpu, std::chrono::steady_clock::time_point now);
    // The badge while the HPC backend is chosen and can run nothing (`why`,
    // core/workbench.hpp's RunGate): red "no engine" in place of "Cluster" or
    // "no worker yet"; a connect under way, a lost or failed one stay as
    // they are.
    ConnectionBadge noEngineBadge(ConnectionBadge badge, const Status& st, const std::string& why);

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

    // --- the worker's launch script -------------------------------------------------
    //
    // app/python/slurm/sirius_worker.sbatch as this application was built
    // with it (cmake/EmbedText.cmake), and the name it is written under in
    // ~/.sirius/run: sirius_worker-<build id>.sbatch.
    const std::string& workerLaunchScript();
    std::string workerLaunchScriptName();
    // The shell lines that write it there (0700, through a temporary file),
    // exposed for tests; they set LS to its path.
    std::string uploadLaunchScript();

    // --- the engine builds ----------------------------------------------------------
    //
    // <Profile::engineBuilds>/<commit>/BUILD.json and bin/sirius-cli, as the
    // checks list them (the newest first), and the one that serves this
    // application: the build of its own commit, else one with the same
    // operations and engine API (engineMismatch). Exposed for tests.
    struct EngineBuild {
        std::string dir;          // the folder's name (a commit)
        bool runnable = false;    // bin/sirius-cli is there and executable
        bool readable = false;    // BUILD.json parses
        bool python = false;      // python/sirius_worker is there: the worker's code of that commit
        BuildInfo info;
    };
    std::string engineBuildsScript(const std::string& folder);
    std::vector<EngineBuild> parseEngineBuilds(const std::string& output);
    // The index of the build to use, or -1; `note` says which and why ("this
    // build", "the same operations as this build"), or why none fits.
    int pickEngineBuild(const std::vector<EngineBuild>& builds, const BuildInfo& app, std::string* note);
    // What to do when no engine build fits (or there is none): build one of
    // this application's commit into `folder` ("" = not set yet), with the
    // command that does it.
    std::string engineBuildFix(const std::string& folder, const BuildInfo& app);
    // What to do when the profile asks for the engine and none is found:
    // set the Engine builds folder; `hint` is a builds folder the checks
    // found on the cluster ("" none).
    std::string noEngineFix(const std::string& hint);

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

        // Step 1: the SSH login (kept when it is up for this host), the
        // partitions, and a job that holds the allocation -- or the job this
        // session left running, reattached to with its worker when that
        // still answers. JobReady (or Connected) afterwards. Ignored while
        // connecting.
        void connectJob(const Profile& profile);
        // Step 2: the worker in the held job: the checks, the srun step, the
        // hello. A worker already running is stopped first (a new image, new
        // data folders). Ignored without a job, or while connecting.
        void startWorker(const Profile& profile);
        // Both steps in one go (scripted runs, tests).
        void connect(const Profile& profile);
        // The SSH login only (kept when it is up for this host), then the
        // cluster's partitions (clusterInfo()): no job is submitted. The
        // session is Idle again afterwards, logged in.
        void logIn(const Profile& profile);
        // Ends the worker step; the job stays held (JobReady). Blocks for as
        // long as scancel takes.
        void stopWorker();
        // Cancels the held job (scancel; its worker goes with it) and keeps
        // the login: Idle, logged in, for a job with other settings. Blocks
        // for as long as scancel takes.
        void cancelJob();
        // Stops a connect in progress (the prompt included).
        void cancelConnect();
        // Closes the worker connection and ssh; with `cancelJob` scancels the
        // job first. Blocks for as long as scancel takes (seconds).
        void disconnect(bool cancelJob);

        // Builds a worker image in the held job: first whether the cluster
        // lets the user build (apptainer build --fakeroot of a tiny image),
        // then `<launcher> build --fakeroot <image> <defFile>`, its output in
        // status().build. Paths are the cluster's. Ignored without a job or
        // while a build runs.
        void buildImage(const Profile& profile, const std::string& defFile, const std::string& image);
        // The probe alone (status().build.supported).
        void probeBuild(const Profile& profile);
        void cancelBuild();

        Status status() const;
        // The profile of the job and of the worker as last started.
        Profile profile() const;
        bool connected() const;   // the worker answers
        bool hasJobRunning() const;   // a job is held (JobReady, Starting or Connected)
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
        enum class Mode { Login,
                          Job,
                          Worker,
                          Both };
        void start(const Profile& profile, Mode mode);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::cluster

#endif // SIRIUS_APP_CLUSTER_HPP
