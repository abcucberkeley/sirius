// The cluster connection (core/remote_host, core/cluster, core/remote_source)
// against a local stand-in: tests/tools/fake_ssh.py plays OpenSSH's ssh
// (prompts through SSH_ASKPASS, a SOCKS5 proxy on -D, `bash -s` as the
// remote shell), tests/tools/fake_slurm plays sbatch / srun / squeue / sacct /
// scancel / sinfo / sacctmgr / scontrol / apptainer: the job holds nothing
// here, its worker step starts the real worker (app/python) or engine on
// 127.0.0.1. No real ssh is run and no host but this one is contacted.
//
// Needs a Python ($SIRIUS_PYTHON, else host::findPython) and bash
// ($SIRIUS_TEST_BASH, else Git's bash on Windows, bash on PATH elsewhere);
// the end-to-end case also numpy in that Python. Cases skip without them.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sirius/tiff_io.hpp>

#include "core/build_info.hpp"
#include "core/cluster.hpp"
#include "core/cluster_profiles.hpp"
#include "core/cluster_wizard.hpp"
#include "core/settings_toml.hpp"
#include "core/errors.hpp"
#include "core/host.hpp"
#include "core/process.hpp"
#include "core/remote_host.hpp"
#include "core/remote_source.hpp"
#include "core/rpc.hpp"
#include "core/workbench.hpp"
#include "temp_path.hpp"

using namespace sirius::app;

namespace {

    namespace fs = std::filesystem;

    void setEnv(const char* name, const std::string& value) {
#ifdef _WIN32
        _putenv_s(name, value.c_str());
#else
        setenv(name, value.c_str(), 1);
#endif
    }

    std::string readAll(const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        std::stringstream s;
        s << in.rdbuf();
        return s.str();
    }

    int countOf(const std::string& text, const std::string& what) {
        int n = 0;
        for (std::size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + what.size())) ++n;
        return n;
    }

    std::string testPython() {
        std::string p = host::environment("SIRIUS_PYTHON");
        if (p.empty()) p = host::findPython();
        return p;
    }

    std::string testBash() {
        std::string b = host::environment("SIRIUS_TEST_BASH");
        if (!b.empty()) return b;
#ifdef _WIN32
        for (const char* p : {"C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"})
            if (host::isFile(p)) return p;
        return {};   // not System32's bash.exe: that is WSL, another machine
#else
        return host::findExecutable("bash");
#endif
    }

    bool pythonHas(const std::string& python, const std::string& module) {
        ChildProcess p;
        ChildProcess::Options o;
        o.program = python;
        o.arguments = {"-c", "import " + module};
        if (!p.start(o)) return false;
        p.closeInput();
        return p.waitForExit(60000) && p.exitCode() == 0;
    }

    // A script the fake cluster runs by name from PATH: written, then made
    // executable (a no-op on Windows, where Git Bash goes by the #! line).
    void writeScript(const fs::path& path, const std::string& text) {
        std::ofstream(path, std::ios::binary) << text;
        std::error_code ec;
        fs::permissions(path, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::add, ec);
    }

    // A temporary "cluster home" with the fake Slurm tools and a python3 shim
    // on its PATH, and the variables fake_ssh.py reads set for the children.
    struct FakeCluster {
        std::string python, bash;
        fs::path root, home, bin, slurm, log;

        bool usable() const { return !python.empty() && !bash.empty(); }

        FakeCluster() : python(testPython()), bash(testBash()) {
            root = sirius::test::uniqueTempPath("cluster", "");
            home = root / "home";
            bin = root / "bin";
            slurm = root / "slurm";
            log = root / "ssh.log";
            fs::create_directories(home);
            fs::create_directories(bin);
            fs::create_directories(slurm);
            // the tools as LF scripts, whatever the checkout's line endings, and
            // executable: a file written here is 0644 on POSIX, and bash finds a
            // non-executable one on PATH all the same, then fails "Permission denied"
            for (const char* tool : {"sbatch", "srun", "squeue", "sacct", "scancel", "sinfo", "sacctmgr", "scontrol", "apptainer"}) {
                std::string text = readAll(fs::path(SIRIUS_TEST_FAKE_SLURM_DIR) / tool);
                std::string lf;
                for (char c : text)
                    if (c != '\r') lf.push_back(c);
                writeScript(bin / tool, lf);
            }
            writeScript(bin / "python3", "#!/bin/bash\nexec \"$FAKE_PYTHON\" \"$@\"\n");
            // the image's python, for the checks in the job (the fake apptainer runs them here)
            writeScript(bin / "python", "#!/bin/bash\nexec \"$FAKE_PYTHON\" \"$@\"\n");
            setEnv("FAKE_SSH_LOG", log.generic_string());
            setEnv("FAKE_SSH_HOME", home.generic_string());
            setEnv("FAKE_SSH_PATH", bin.string());
            setEnv("FAKE_SSH_BASH", bash);
            setEnv("FAKE_SSH_USER", "tester");
            setEnv("FAKE_PYTHON", python);
            setEnv("FAKE_SLURM_DIR", slurm.generic_string());
            setEnv("FAKE_SLURM_KILL_ON_EXIT", "1");
            setEnv("FAKE_SLURM_PENDING_POLLS", "2");
            setEnv("FAKE_SSH_PROMPTS", "[]");
            setEnv("FAKE_SSH_ANSWERS", "[]");
            setEnv("FAKE_SINFO_FAIL", "0");
            setEnv("FAKE_SACCTMGR_FAIL", "0");
            setEnv("FAKE_SCONTROL_FAIL", "0");
            setEnv("USER", "tester");
            setEnv("FAKE_CONTAINER_SITE", "");
            setEnv("FAKE_APPTAINER_DRY", "0");
            setEnv("FAKE_APPTAINER_NO_FAKEROOT", "0");
            setEnv("SIRIUS_WORKBENCH_PY", std::string(SIRIUS_TEST_SOURCE_DIR) + "/bindings/python/sirius/workbench.py");
        }
        ~FakeCluster() {
            std::error_code ec;
            fs::remove_all(root, ec);
        }
        void prompts(const std::string& prompts, const std::string& answers) {
            setEnv("FAKE_SSH_PROMPTS", prompts);
            setEnv("FAKE_SSH_ANSWERS", answers);
        }
        ssh::Options options() const {
            ssh::Options o;
            o.program = python;
            o.programArgs = {SIRIUS_TEST_FAKE_SSH};
            o.host = "fakecluster";
            return o;
        }
        std::string sshLog() const { return readAll(log); }
        // An nvidia-smi on the cluster's PATH that lists `lines` (its
        // --query-gpu=index,name,memory.total,uuid CSV): the job's GPU
        // hardware, whatever CUDA the worker's Python has.
        void fakeNvidiaSmi(const std::vector<std::string>& lines) const {
            std::string sh = "#!/bin/sh\n", bat = "@echo off\r\n";
            for (const std::string& l : lines) {
                sh += "echo '" + l + "'\n";
                bat += "echo " + l + "\r\n";
            }
            writeScript(bin / "nvidia-smi", sh);
            std::ofstream(bin / "nvidia-smi.bat", std::ios::binary) << bat;   // what a Windows Python finds
            // a test run inside a Slurm job: the fake GPU is the job's own
            if (!host::environment("SLURM_JOB_ID").empty()) setEnv("SLURM_JOB_GPUS", "0");
        }
    };

    // A .npy file of uint16 (c, t, z, y, x) = values i % 997.
    void writeNpy(const fs::path& p, int c, int t, int z, int y, int x) {
        std::string header = "{'descr': '<u2', 'fortran_order': False, 'shape': (" + std::to_string(c) + ", " + std::to_string(t) + ", " +
                             std::to_string(z) + ", " + std::to_string(y) + ", " + std::to_string(x) + "), }";
        while ((10 + header.size() + 1) % 64 != 0) header.push_back(' ');
        header.push_back('\n');
        std::ofstream out(p, std::ios::binary);
        out << "\x93NUMPY" << static_cast<char>(1) << static_cast<char>(0);
        const std::uint16_t len = static_cast<std::uint16_t>(header.size());
        out.put(static_cast<char>(len & 0xff));
        out.put(static_cast<char>(len >> 8));
        out << header;
        const long long n = 1LL * c * t * z * y * x;
        for (long long i = 0; i < n; ++i) {
            const std::uint16_t v = static_cast<std::uint16_t>(i % 997);
            out.put(static_cast<char>(v & 0xff));
            out.put(static_cast<char>(v >> 8));
        }
    }

    template <typename F> bool waitFor(F done, std::chrono::seconds limit) {
        const auto end = std::chrono::steady_clock::now() + limit;
        while (std::chrono::steady_clock::now() < end) {
            if (done()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return done();
    }

    // Neither getting the job nor starting the worker.
    bool settled(const cluster::Session& s) {
        const cluster::State st = s.status().state;
        return st != cluster::State::Connecting && st != cluster::State::Starting;
    }

    // A stand-in for the image's packages: a sirius that imports.
    fs::path imageSite(const fs::path& root) {
        const fs::path site = root / "image-site";
        fs::create_directories(site / "sirius");
        std::ofstream(site / "sirius" / "__init__.py") << "__version__ = '0-test'\n";
        return site;
    }

} // namespace

TEST_CASE("cluster: shell words and the ssh command line", "[app][cluster]") {
    CHECK(ssh::shellQuote("a b") == "'a b'");
    CHECK(ssh::shellQuote("it's") == "'it'\\''s'");
    CHECK(ssh::remotePathWord("~/dev/sirius") == "\"$HOME\"/'dev/sirius'");
    CHECK(ssh::remotePathWord("~") == "\"$HOME\"");
    CHECK(ssh::remotePathWord("/scratch/x y") == "'/scratch/x y'");
    ssh::Options o;
    o.host = "fiona";
    const std::vector<std::string> a = ssh::sshArguments(o, 40000);
    auto has = [&](const std::string& opt) {
        for (std::size_t i = 0; i + 1 < a.size(); ++i)
            if (a[i] == "-o" && a[i + 1] == opt) return true;
        return false;
    };
    // a wrong password costs one attempt; prompts go through askpass
    CHECK(has("NumberOfPasswordPrompts=1"));
    CHECK(has("BatchMode=no"));
    CHECK(has("ConnectTimeout=15"));
    CHECK(has("ExitOnForwardFailure=yes"));
    REQUIRE(a.size() >= 3);
    CHECK(a[a.size() - 2] == "fiona");
    CHECK(a[a.size() - 3] == "--");
    bool socks = false;
    for (std::size_t i = 0; i + 1 < a.size(); ++i) socks = socks || (a[i] == "-D" && a[i + 1] == "127.0.0.1:40000");
    CHECK(socks);
    // the cluster gets neither this machine's display nor its agent, and no
    // LocalCommand of the user's ssh config runs here; forwardings are not
    // cleared, since -D is one
    const auto flag = [&](const std::string& f) { return std::find(a.begin(), a.end(), f) != a.end(); };
    CHECK(flag("-x"));
    CHECK(flag("-a"));
    CHECK(has("ForwardAgent=no"));
    CHECK(has("ForwardX11=no"));
    CHECK(has("PermitLocalCommand=no"));
    CHECK(has("StrictHostKeyChecking=yes"));
    for (const std::string& s : a) CHECK(s.find("ClearAllForwardings") == std::string::npos);
}

TEST_CASE("cluster: only OpenSSH's host key question is answered in clear", "[app][cluster]") {
    CHECK(ssh::isHostKeyConfirmation("confirm", "anything"));
    CHECK(ssh::isHostKeyConfirmation("", "The authenticity of host 'fiona (10.0.0.1)' can't be established.\n"
                                         "ED25519 key fingerprint is SHA256:abc.\n"
                                         "Are you sure you want to continue connecting (yes/no/[fingerprint])? "));
    // a server's keyboard-interactive prompt, whatever it says
    CHECK_FALSE(ssh::isHostKeyConfirmation("", "(tester@fiona) Password (yes/no): "));
    CHECK_FALSE(ssh::isHostKeyConfirmation("", "(tester@fiona) Are you sure you want to continue connecting (yes/no)? "));
    CHECK_FALSE(ssh::isHostKeyConfirmation("", "Password: "));
}

TEST_CASE("cluster: the askpass relay drops a silent peer and stops listening after the login", "[app][cluster]") {
    std::atomic<int> asked{0};
    ssh::AskpassServer relay([&](const ssh::Prompt&) -> std::optional<std::string> {
        ++asked;
        return std::string("x");
    });
    const auto env = relay.environment("helper");
    bool display = false;
    for (const auto& kv : env) display = display || kv.first == "DISPLAY";
#ifdef _WIN32
    CHECK_FALSE(display);   // Windows' OpenSSH needs none for SSH_ASKPASS_REQUIRE=force
#else
    CHECK(display);         // OpenSSH before 8.4 wants one
#endif
    // A peer that connects and says nothing is dropped after a second, not
    // after five, and holds no prompt up meanwhile.
    std::unique_ptr<rpc::Transport> silent = rpc::connectTcp("127.0.0.1", relay.port(), std::chrono::seconds(5));
    std::unique_ptr<rpc::Transport> second = rpc::connectTcp("127.0.0.1", relay.port(), std::chrono::seconds(5));
    const auto t0 = std::chrono::steady_clock::now();
    bool closed = false;
    std::vector<std::byte> in;
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(6)) {
        try {
            silent->receive(in, std::chrono::milliseconds(100));
        } catch (const ProtocolError&) {
            closed = true;
            break;
        }
    }
    CHECK(closed);
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(3));
    CHECK(asked.load() == 0);   // without the secret nothing reaches the handler
    relay.close();
    CHECK_THROWS_AS(rpc::connectTcp("127.0.0.1", relay.port(), std::chrono::seconds(3)), ProtocolError);
}

TEST_CASE("cluster: a listing comes back folders first, names as they are", "[app][cluster]") {
    const cluster::Listing l = cluster::parseListing(
        R"({"path": "/home/u", "home": "/home/u", "truncated": true, "entries": [["b.tif", 0, 10, 5, 0], ["Zeta", 1, 0, 0, 0], ["a dir", 1, 0, 0, 1], ["caf\u00e9.tif", 0, 3, 0, 0]]})");
    CHECK(l.path == "/home/u");
    CHECK(l.truncated);
    REQUIRE(l.entries.size() == 4);
    CHECK(l.entries[0].name == "a dir");
    CHECK(l.entries[0].link);
    CHECK(l.entries[1].name == "Zeta");
    CHECK(l.entries[2].name == "b.tif");
    CHECK(l.entries[2].size == 10);
    CHECK(l.entries[3].name == "caf\xC3\xA9.tif");
    CHECK_THROWS_AS(cluster::parseListing(R"({"error": "/x: Permission denied"})"), ssh::SshError);
    std::string host, path;
    CHECK(splitClusterPath(makeClusterPath("fiona", "/home/u/a b.tif"), host, path));
    CHECK(host == "fiona");
    CHECK(path == "/home/u/a b.tif");
    CHECK(isRemoteDatasetPath("cluster://fiona/x.tif"));
    CHECK_FALSE(isRemoteDatasetPath("C:/data/x.tif"));
}

TEST_CASE("cluster: the command channel frames each command's output and exit code", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    ssh::Session s;
    s.open(fc.options(), {}, std::chrono::seconds(60));
    REQUIRE(s.isOpen());
    CHECK(s.socksPort() > 0);

    ssh::CommandResult r = s.run("echo hi; echo there; echo oops >&2; exit 3");
    CHECK(r.out == "hi\nthere");
    CHECK(r.err == "oops");
    CHECK(r.exitCode == 3);
    // an `exit` ended only that command, and output without a newline is whole
    r = s.run("printf 'no newline'");
    CHECK(r.out == "no newline");
    CHECK(r.ok());
    // a syntax error is the command's failure, not the session's end
    r = s.run("if then fi (");
    CHECK_FALSE(r.ok());
    CHECK_FALSE(r.err.empty());
    r = s.run("echo 'caf\xC3\xA9 \xE2\x9C\x93'; read x; echo \"[$x]\"");
    CHECK(r.out == "caf\xC3\xA9 \xE2\x9C\x93\n[]");   // stdin is /dev/null, not the next command
    CHECK(s.run("echo alive").out == "alive");

    // a listing of the cluster's home
    fs::create_directories(fc.home / "a folder");
    std::ofstream(fc.home / "stack 1.tif") << "x";
    std::ofstream(fc.home / "caf\xC3\xA9.npy") << "xy";
    r = s.run(cluster::listingScript("~", 100));
    REQUIRE(r.ok());
    const cluster::Listing l = cluster::parseListing(r.out.substr(r.out.find('{')));
    REQUIRE(l.entries.size() >= 3);
    CHECK(l.entries[0].name == "a folder");
    CHECK(l.entries[0].dir);
    bool unicode = false, spaces = false;
    for (const auto& e : l.entries) {
        unicode = unicode || e.name == "caf\xC3\xA9.npy";
        spaces = spaces || (e.name == "stack 1.tif" && e.size == 1);
    }
    CHECK(unicode);
    CHECK(spaces);
    // the cap
    const cluster::Listing capped = cluster::parseListing(s.run(cluster::listingScript("~", 1)).out);
    CHECK(capped.entries.size() == 1);
    CHECK(capped.truncated);
    s.close();
    CHECK_FALSE(s.isOpen());
    CHECK_THROWS_AS(s.run("echo x"), ssh::SshError);
}

TEST_CASE("cluster: ssh's prompts reach the application through the askpass relay", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    // the second prompt's "(yes/no)" is the server's text: still a secret
    fc.prompts(R"(["Password: ", "Verification code (yes/no): "])", R"(["hunter2", "123456"])");
    std::vector<std::string> asked;
    ssh::AskpassServer relay([&](const ssh::Prompt& p) -> std::optional<std::string> {
        asked.push_back(p.text);
        CHECK_FALSE(p.echo);
        return asked.size() == 1 ? std::string("hunter2") : std::string("123456");
    });
    ssh::Options o = fc.options();
    o.environment = relay.environment(SIRIUS_TEST_ASKPASS);
    // a local process that connects to the relay and says nothing holds up
    // neither the prompts nor the login
    std::unique_ptr<rpc::Transport> silent = rpc::connectTcp("127.0.0.1", relay.port(), std::chrono::seconds(5));
    ssh::Session s;
    s.open(o, {}, std::chrono::seconds(60));
    REQUIRE(asked.size() == 2);
    CHECK(asked[0] == "(tester@fakecluster) Password: ");
    CHECK(asked[1] == "(tester@fakecluster) Verification code (yes/no): ");
    CHECK(fc.sshLog().find("\"ForwardAgent=no\"") != std::string::npos);   // what ssh was really started with
    CHECK(s.run("echo in").out == "in");
    const std::string log = fc.sshLog();
    CHECK(countOf(log, "response ok") == 2);
    CHECK(log.find("NumberOfPasswordPrompts=1") != std::string::npos);
    // the answers went to ssh's helper only: not in its arguments, not in the log
    CHECK(log.find("hunter2") == std::string::npos);
    CHECK(log.find("123456") == std::string::npos);
}

TEST_CASE("cluster: a command that does not answer closes the session", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    ssh::Session s;
    s.open(fc.options(), {}, std::chrono::seconds(60));
    REQUIRE(s.isOpen());
    CHECK_THROWS_AS(s.run("sleep 30", std::chrono::milliseconds(500)), ssh::SshError);
    CHECK_FALSE(s.isOpen());
    CHECK_THROWS_WITH(s.run("echo hi", std::chrono::seconds(5)), Catch::Matchers::ContainsSubstring("closed"));
}

TEST_CASE("cluster: cancelling a command closes the session", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    ssh::Session s;
    s.open(fc.options(), {}, std::chrono::seconds(60));
    REQUIRE(s.isOpen());
    std::atomic<bool> stop{false};
    std::thread later([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        stop.store(true);
    });
    CHECK_THROWS_AS(s.run("sleep 30", std::chrono::seconds(10), [&] { return stop.load(); }), ssh::SshError);
    later.join();
    CHECK_FALSE(s.isOpen());
    CHECK_THROWS_WITH(s.run("echo hi", std::chrono::seconds(5)), Catch::Matchers::ContainsSubstring("closed"));
}

TEST_CASE("cluster: a wrong password is one attempt, reported, never retried", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    fc.prompts(R"(["Password: "])", R"(["right"])");
    int asked = 0;
    ssh::AskpassServer relay([&](const ssh::Prompt&) -> std::optional<std::string> {
        ++asked;
        return std::string("wrong");
    });
    ssh::Options o = fc.options();
    o.environment = relay.environment(SIRIUS_TEST_ASKPASS);
    ssh::Session s;
    try {
        s.open(o, {}, std::chrono::seconds(60));
        FAIL("the login should have failed");
    } catch (const ssh::SshError& e) {
        CHECK(std::string(e.detail).find("Permission denied") != std::string::npos);
    }
    CHECK(asked == 1);
    CHECK(countOf(fc.sshLog(), "argv ") == 1);   // ssh started once
    CHECK_FALSE(s.isOpen());
}

TEST_CASE("cluster: a cancelled prompt stops ssh before its helper answers", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    fc.prompts(R"(["Password: "])", R"(["right"])");
    cluster::Session session;
    session.setAskpassProgram(SIRIUS_TEST_ASKPASS);
    int asked = 0;
    session.setPrompt([&](const ssh::Prompt&) -> std::optional<std::string> {
        ++asked;
        return std::nullopt;   // the user pressed Cancel
    });
    cluster::Profile p;
    p.host = "fakecluster";
    p.sshProgram = fc.python;
    p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
    session.connect(p);
    REQUIRE(waitFor([&] { return session.status().state == cluster::State::Disconnected; }, std::chrono::seconds(60)));
    const cluster::Status st = session.status();
    CHECK(st.reason.find("cancelled") != std::string::npos);
    CHECK(st.steps[0].status == cluster::StepStatus::Failed);
    CHECK(asked == 1);
    // let a helper that outlived ssh finish, then read what ssh saw
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const std::string log = fc.sshLog();
    CHECK(countOf(log, "asking ") == 1);
    CHECK(log.find("response") == std::string::npos);   // nothing, not even an empty answer, was sent
    CHECK(log.find("answered") == std::string::npos);
}

// The worker describes each array it sends: the description's shape sizes
// the output, so it is checked before it allocates anything, and compressed
// bytes may inflate to that size and not one byte past it.
TEST_CASE("cluster: an array description from the worker cannot size an allocation or inflate past itself", "[app][cluster]") {
    using json = nlohmann::json;
    std::vector<sirius::Index> shape;
    rpc::Tensor data;
    data.bytes.assign(16, std::byte{0});
    // a shape whose product wraps a 64-bit count
    CHECK_THROWS_AS(decodeWorkerArray(json{{"shape", {1ll << 32, 1ll << 32, 16}}, {"dtype", "uint8"}}, data, shape), ProtocolError);
    // a compressed array claiming far more than its bytes could inflate to
    CHECK_THROWS_AS(decodeWorkerArray(json{{"shape", {1 << 20, 1 << 10}}, {"dtype", "float32"}, {"encoding", "zlib"}}, data, shape),
                    ProtocolError);
    // 4096 zero bytes, deflated to 26: described as 16 bytes, they must not be followed past 16
    const unsigned char bomb[] = {0x78, 0xda, 0xed, 0xc1, 0x01, 0x0d, 0x00, 0x00, 0x00, 0xc2, 0xa0, 0xf7, 0x4f,
                                  0x6d, 0x0f, 0x07, 0x14, 0x00, 0x00, 0x00, 0xf0, 0x6e, 0x10, 0x00, 0x00, 0x01};
    rpc::Tensor packed;
    for (unsigned char b : bomb) packed.bytes.push_back(static_cast<std::byte>(b));
    CHECK_THROWS_AS(decodeWorkerArray(json{{"shape", {16}}, {"dtype", "uint8"}, {"encoding", "zlib"}}, packed, shape), ProtocolError);
    // and described as what it is, it decodes
    const std::vector<float> v = decodeWorkerArray(json{{"shape", {64, 64}}, {"dtype", "uint8"}, {"encoding", "zlib"}}, packed, shape);
    CHECK(v.size() == 4096);
    const std::vector<sirius::Index> want{64, 64};
    CHECK(shape == want);
}

TEST_CASE("cluster: SOCKS through the ssh proxy says when nothing listens", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    ssh::Session s;
    s.open(fc.options(), {}, std::chrono::seconds(60));
    const int nobody = ssh::freeLocalPort();
    try {
        rpc::connectSocks5("127.0.0.1", s.socksPort(), "node42", nobody, std::chrono::seconds(5));
        FAIL("nothing listens there");
    } catch (const ProtocolError& e) {
        CHECK(std::string(e.what()).find("node42:" + std::to_string(nobody)) != std::string::npos);
    }
    CHECK(fc.sshLog().find("socks node42:") != std::string::npos);   // the name went to the far side unresolved
    s.close();
    CHECK_THROWS_AS(rpc::connectSocks5("127.0.0.1", s.socksPort(), "node42", 1, std::chrono::seconds(2)), ProtocolError);
}

TEST_CASE("cluster: the node's GPU and CPU are named, and a GPU the worker cannot use says why", "[app][cluster]") {
    const std::string noLibrary = "no CUDA library in the worker's environment: install torch or build the sirius package with CUDA";
    // the user's job: an A100 on g0003, a venv with neither torch nor the sirius package
    WorkerCapabilities caps;
    caps.device = "cpu \xC2\xB7 16 threads";
    parseWorkerHardware({{"cuda", false},
                         {"gpus", nlohmann::json::array({{{"name", "NVIDIA A100-SXM4-80GB"}, {"memory_mb", 81920}}})},
                         {"cuda_usable", false},
                         {"cuda_reason", noLibrary},
                         {"cpu_threads", 16}},
                        caps);
    REQUIRE(caps.gpus.size() == 1);
    CHECK(caps.gpus[0].name == "NVIDIA A100-SXM4-80GB");
    CHECK(caps.gpus[0].memoryMb == 81920);
    CHECK_FALSE(caps.cudaUsable);
    CHECK(caps.cudaReason == noLibrary);
    CHECK(caps.cpuThreads == 16);
    CHECK_FALSE(cluster::gpuUsable(caps));
    CHECK(cluster::gpuSummary(caps.gpus) == "1\xC3\x97 A100 80 GB");
    CHECK(cluster::gpuUnusableReason("g0003.abc0", caps) == "g0003.abc0 has 1\xC3\x97 A100 80 GB, but the worker cannot compute on it: " + noLibrary);
    CHECK(cluster::unusableGpuNote(caps) == "1\xC3\x97 A100 80 GB not usable: " + noLibrary);
    std::vector<cluster::NodeDevice> d = cluster::nodeDevices("g0003.abc0", caps);
    REQUIRE(d.size() == 2);
    CHECK(d[0].gpu);
    CHECK(d[0].label == "g0003 \xC2\xB7 1\xC3\x97 A100 80 GB");
    CHECK_FALSE(d[0].usable);
    CHECK(d[0].why == noLibrary);
    CHECK_FALSE(d[1].gpu);
    CHECK(d[1].label == "g0003 \xC2\xB7 CPU \xC2\xB7 16 threads");
    CHECK(d[1].usable);

    // the same GPU with CUDA in the worker: usable, nothing to say
    WorkerCapabilities gpu;
    gpu.cuda = true;
    gpu.device = "cuda:0 \xC2\xB7 NVIDIA A100-SXM4-80GB \xC2\xB7 80 GB";
    parseWorkerHardware({{"gpus", nlohmann::json::array({{{"name", "NVIDIA A100-SXM4-80GB"}, {"memory_mb", 81920}}})}, {"cuda_usable", true}, {"cuda_reason", ""}}, gpu);
    CHECK(cluster::gpuUsable(gpu));
    CHECK(cluster::gpuUnusableReason("g0003", gpu).empty());
    CHECK(cluster::unusableGpuNote(gpu).empty());
    CHECK(cluster::nodeDevices("g0003", gpu)[0].usable);

    // a worker of protocol 2 without the new fields: usable as its "cuda" says
    WorkerCapabilities old;
    old.cuda = true;
    old.device = "cuda:0 \xC2\xB7 RTX 4000 \xC2\xB7 20 GB";
    parseWorkerHardware({{"cuda", true}, {"device", old.device}}, old);
    CHECK(old.cudaUsable);
    CHECK(old.gpus.empty());
    CHECK(old.cpuThreads == 0);
    d = cluster::nodeDevices("n0042", old);
    CHECK(d[0].label == "n0042 \xC2\xB7 cuda:0 \xC2\xB7 RTX 4000 \xC2\xB7 20 GB");
    CHECK(d[0].usable);
    CHECK(d[1].label == "n0042 \xC2\xB7 CPU");

    // a job without a GPU
    WorkerCapabilities none;
    none.device = "cpu \xC2\xB7 8 threads";
    parseWorkerHardware({{"gpus", nlohmann::json::array()}, {"cuda_usable", false}, {"cuda_reason", "this worker job has no GPU"}, {"cpu_threads", 8}}, none);
    CHECK(cluster::gpuUnusableReason("n0042", none) ==
          "The worker job on n0042 has no GPU (it reports cpu \xC2\xB7 8 threads): reconnect with GPUs \xE2\x89\xA5 1 to use one");
    CHECK(cluster::unusableGpuNote(none).empty());
    d = cluster::nodeDevices("n0042", none);
    CHECK(d[0].label == "n0042 \xC2\xB7 no GPU");
    CHECK_FALSE(d[0].usable);
    CHECK(d[0].why == "this worker job has no GPU");
    CHECK(d[1].label == "n0042 \xC2\xB7 CPU \xC2\xB7 8 threads");

    // names and counts
    CHECK(cluster::shortGpuName("NVIDIA A100-SXM4-80GB") == "A100");
    CHECK(cluster::shortGpuName("NVIDIA A100 80GB PCIe") == "A100");
    CHECK(cluster::shortGpuName("NVIDIA H100 80GB HBM3") == "H100");
    CHECK(cluster::shortGpuName("Tesla V100-SXM2-32GB") == "V100");
    CHECK(cluster::shortGpuName("NVIDIA GeForce RTX 4090") == "GeForce RTX 4090");
    CHECK(cluster::shortGpuName("NVIDIA RTX 4000 Ada Generation") == "RTX 4000 Ada Generation");
    CHECK(cluster::gpuSummary({{"NVIDIA A100-SXM4-80GB", 81920}, {"NVIDIA A100-SXM4-80GB", 81920}, {"Tesla V100-SXM2-32GB", 32768}}) ==
          "2\xC3\x97 A100 80 GB + 1\xC3\x97 V100 32 GB");
    CHECK(cluster::gpuSummary({{"NVIDIA A100-PCIE-40GB", 40960}}) == "1\xC3\x97 A100 40 GB");
    CHECK(cluster::gpuSummary({}).empty());
    CHECK(cluster::shortNodeName("g0003.abc0") == "g0003");
    CHECK(cluster::shortNodeName("n0042") == "n0042");
    CHECK(cluster::shortNodeName("10.0.0.5") == "10.0.0.5");

    // what is not of the expected form is left out, never thrown on
    WorkerCapabilities odd;
    parseWorkerHardware({{"gpus", nlohmann::json::array({nlohmann::json("A100"), nlohmann::json{{"name", 7}, {"memory_mb", "80 GB"}}})}, {"cuda_usable", "yes"}, {"cpu_threads", -3}}, odd);
    REQUIRE(odd.gpus.size() == 1);
    CHECK(odd.gpus[0].name.empty());
    CHECK(odd.gpus[0].memoryMb == 0);
    CHECK_FALSE(odd.cudaUsable);
    CHECK(odd.cpuThreads == 0);
    parseWorkerHardware({{"gpus", "A100"}}, odd);
    CHECK(odd.gpus.size() == 1);
}

TEST_CASE("cluster: connect submits the worker, waits, says hello and serves a dataset", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the worker");
    // a checkout on the "cluster": the worker package and the job template
    const fs::path checkout = fc.home / "sirius";
    fs::create_directories(checkout / "app" / "python");
    for (const char* d : {"sirius_worker", "slurm"})
        fs::copy(fs::path(SIRIUS_TEST_SOURCE_DIR) / "app" / "python" / d, checkout / "app" / "python" / d,
                 fs::copy_options::recursive | fs::copy_options::skip_existing);
    writeNpy(fc.home / "stack.npy", 1, 2, 4, 64, 80);
    fc.prompts(R"(["Password: "])", R"(["pw"])");
    // the job's GPU, which the test's Python has no CUDA for (or has)
    fc.fakeNvidiaSmi({"0, NVIDIA A100-SXM4-80GB, 81920, GPU-fake-0001"});

    cluster::Session session;
    session.setAskpassProgram(SIRIUS_TEST_ASKPASS);
    session.setPrompt([](const ssh::Prompt&) -> std::optional<std::string> { return std::string("pw"); });
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    std::vector<std::string> logLines;
    std::mutex logMutex;
    session.setLog([&](const std::string& l) {
        const std::lock_guard<std::mutex> g(logMutex);
        logLines.push_back(l);
    });
    std::ofstream(fc.home / "w.sif") << "image\n";
    setEnv("FAKE_CONTAINER_SITE", imageSite(fc.root).string());
    cluster::Profile p;
    p.host = "fakecluster";
    p.sshProgram = fc.python;
    p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
    p.checkout = "~/sirius";
    p.container = "~/w.sif";
    p.engine = false;
    p.partition = "abc_a100";
    p.port = ssh::freeLocalPort();
    bool sawPending = false;
    session.setChanged([&] {
        const cluster::Status st = session.status();
        if (st.steps[static_cast<int>(cluster::Step::Queue)].detail.find("PENDING") != std::string::npos) sawPending = true;
    });
    session.connect(p);
    REQUIRE(waitFor([&] { return settled(session); }, std::chrono::seconds(180)));
    cluster::Status st = session.status();
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    CHECK(sawPending);
    CHECK(st.node == "fakenode");
    CHECK(st.jobId == "4711");
    CHECK(st.caps.protocolVersion == rpc::kProtocolVersion);
    CHECK_FALSE(st.caps.version.empty());
    // hello names the GPU from nvidia-smi; a worker without CUDA says why
    // it cannot compute there, and the Hello step and the log say so too
    REQUIRE(st.caps.gpus.size() == 1);
    CHECK(st.caps.gpus[0].name == "NVIDIA A100-SXM4-80GB");
    CHECK(st.caps.gpus[0].memoryMb == 81920);
    CHECK(st.caps.cpuThreads >= 1);
    if (!st.caps.cudaUsable) {
        CHECK_FALSE(st.caps.cudaReason.empty());
        const cluster::StepState& hello = st.steps[static_cast<int>(cluster::Step::Hello)];
        CHECK(hello.status == cluster::StepStatus::Warning);
        CHECK(hello.detail.find("1\xC3\x97 A100 80 GB not usable: " + st.caps.cudaReason) != std::string::npos);
        const std::lock_guard<std::mutex> g(logMutex);
        bool said = false;
        for (const std::string& l : logLines) said = said || (l.find("HPC: connected") != std::string::npos && l.find("A100 80 GB not usable") != std::string::npos);
        CHECK(said);
    }
    for (const auto& step : st.steps) CHECK((step.status == cluster::StepStatus::Done || step.status == cluster::StepStatus::Warning));
    // The token reached the job as a file the worker read and deleted: never
    // its command line, never its environment (Slurm's accounting may keep
    // that), and the job's log is in the private ~/.sirius/run.
    const std::string args = readAll(fc.slurm / "4711.args");
    const std::string env = readAll(fc.slurm / "4711.env");
    CHECK(env.find("SIRIUS_TOKEN_FILE=<set>") != std::string::npos);
    CHECK(env.find("SIRIUS_TOKEN=<set>") == std::string::npos);
    CHECK(env.find("SIRIUS_PORT=0") != std::string::npos);
    const fs::path run = fc.home / ".sirius" / "run";
    CHECK(fs::exists(run / "sirius-job-4711.log"));       // the job's
    CHECK(fs::exists(run / "sirius-worker-4711-1.log"));  // its first worker step's
    for (const auto& entry : fs::directory_iterator(run)) CHECK(entry.path().filename().string().rfind("token.", 0) != 0);
    CHECK(readAll(run / "sirius-worker-4711-1.log").find(session.endpoint().token) == std::string::npos);
#ifndef _WIN32
    CHECK((fs::status(run).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);
    CHECK(readAll(fc.slurm / "4711.tokenmode").rfind("600", 0) == 0);
    CHECK(readAll(fc.slurm / "4711.umask").rfind("0077", 0) == 0);
#endif
    CHECK(session.endpoint().port > 0);
    CHECK(args.find("--parsable") != std::string::npos);
    CHECK(args.find("--partition=abc_a100") != std::string::npos);
    CHECK(args.find("--wrap=") != std::string::npos);   // the job only holds the node
    CHECK(args.find(session.endpoint().token) == std::string::npos);
    CHECK(fc.sshLog().find(session.endpoint().token) == std::string::npos);
    // the worker: a step of that job
    const std::string steps = readAll(fc.slurm / "srun.args");
    CHECK(steps.find("--jobid=4711 --overlap") != std::string::npos);
    CHECK(steps.find(cluster::workerLaunchScriptName()) != std::string::npos);
    CHECK(steps.find(session.endpoint().token) == std::string::npos);

    // the cluster's file system
    const cluster::Listing l = session.list("~");
    bool found = false;
    for (const auto& e : l.entries) found = found || e.name == "stack.npy";
    CHECK(found);

    // a dataset that stays there: meta, a display-sized view, a full plane
    auto datasets = std::make_shared<RemoteDatasets>("fakecluster", [&] { return session.connectWorker(); });
    datasets->install();
    const std::string name = makeClusterPath("fakecluster", l.path + "/stack.npy");
    const DatasetMeta meta = probeDataset(name);
    CHECK(meta.dims.c == 1);
    CHECK(meta.dims.t == 2);
    CHECK(meta.dims.z == 4);
    CHECK(meta.dims.y == 64);
    CHECK(meta.dims.x == 80);
    OpenResult opened = openDataset(name);
    REQUIRE(opened.source);
    ViewProvider* views = opened.source->viewProvider();
    REQUIRE(views);
    ViewRequest req;
    req.kind = ViewRequest::Kind::XY;
    req.t = 1;
    req.index = 2;
    req.factor = 2;
    bool exact = true;
    CHECK_FALSE(views->view(req, exact));   // nothing yet: queued
    auto* remote = dynamic_cast<RemoteSource*>(opened.source.get());
    REQUIRE(remote);
    remote->waitIdle();
    auto tile = views->view(req, exact);
    INFO(views->lastError());
    REQUIRE(tile);
    CHECK(exact);
    CHECK(tile->w == 40);
    CHECK(tile->h == 32);
    // the block mean of the 2 x 2 voxels at (y 0, x 0) of (c 0, t 1, z 2)
    const long long base = (1LL * 4 + 2) * 64 * 80;
    const double mean = ((base % 997) + ((base + 1) % 997) + ((base + 80) % 997) + ((base + 81) % 997)) / 4.0;
    CHECK(std::abs(tile->data[0] - static_cast<float>(mean)) <= 0.5f);   // rounded to uint16 on the worker
    // the neighbouring planes came along behind it
    ViewRequest next = req;
    next.index = 3;
    remote->waitIdle();
    views->view(next, exact);
    CHECK(exact);
    std::vector<float> plane(64 * 80);
    opened.source->readPlane(0, 1, 2, plane.data());
    CHECK(plane[81] == static_cast<float>((base + 81) % 997));
    CHECK(remote->inputReference(0, 1).value("path", std::string()) == l.path + "/stack.npy");
    const TransferStats ts = datasets->stats();
    CHECK(ts.requests >= 3);
    datasets->uninstall();
    opened.source.reset();

    // the keep-alive: still connected after a few pings
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    CHECK(session.connected());

    session.disconnect(true);
    CHECK(session.status().state == cluster::State::Disconnected);
    CHECK(readAll(fc.slurm / "cancelled").find("4711") != std::string::npos);
    CHECK_FALSE(session.sshUp());
    setEnv("FAKE_CONTAINER_SITE", "");
}

TEST_CASE("cluster: a job that dies is noticed and named", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the worker");
    const fs::path checkout = fc.home / "sirius";
    fs::create_directories(checkout / "app" / "python");
    for (const char* d : {"sirius_worker", "slurm"})
        fs::copy(fs::path(SIRIUS_TEST_SOURCE_DIR) / "app" / "python" / d, checkout / "app" / "python" / d,
                 fs::copy_options::recursive | fs::copy_options::skip_existing);
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(300));
    std::ofstream(fc.home / "w.sif") << "image\n";
    setEnv("FAKE_CONTAINER_SITE", imageSite(fc.root).string());
    cluster::Profile p;
    p.host = "fakecluster";
    p.sshProgram = fc.python;
    p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
    p.checkout = "~/sirius";
    p.container = "~/w.sif";
    p.engine = false;
    p.gpus = 0;
    p.port = ssh::freeLocalPort();
    session.connect(p);
    REQUIRE(waitFor([&] { return settled(session); }, std::chrono::seconds(180)));
    INFO(session.status().reason);
    REQUIRE(session.connected());
    // the job ends behind the application's back (a wall-time limit, an admin)
    REQUIRE(session.run("scancel 4711").ok());
    REQUIRE(waitFor([&] { return !session.connected(); }, std::chrono::seconds(60)));
    const cluster::Status st = session.status();
    CHECK(st.reason.find("4711") != std::string::npos);
    CHECK(st.reason.find("CANCELLED") != std::string::npos);
    CHECK(st.sshUp);   // the login is kept: browsing goes on, Connect submits anew
    CHECK(st.dropped);
    CHECK(cluster::connectionBadge(st, false, std::chrono::steady_clock::now()).label == "Cluster: lost");
    session.disconnect(false);
    setEnv("FAKE_CONTAINER_SITE", "");
}

// --- the partitions -----------------------------------------------------------------------

namespace {
    // What clusterInfoScript() prints on a cluster like fiona.
    const char* kInfoOutput = "@@user tester\n"
                              "@@sinfo\n"
                              "cpu*|up|7-00:00:00|6|idle|(null)|64|257000\n"
                              "cpu*|up|7-00:00:00|4|alloc|(null)|64|257000\n"
                              "abc_a100|up|3-00:00:00|2|idle|gpu:a100:1(S:0)|32|500000\n"
                              "abc_a100|up|3-00:00:00|1|mix|gpu:a100:1(S:0)|32|500000\n"
                              "abc_a100|up|3-00:00:00|1|drain*|gpu:a100:1(S:0)|32|500000\n"
                              "dgx|up|1-00:00:00|1|mix|gpu:a100:8(S:0-1)|256|1000000\n"
                              "lab_h100|up|2-00:00:00|2|alloc|gpu:h100:4(S:0,1),shard:h100:16|96+|750000\n"
                              "a line sinfo would never print\n"
                              "@@rc 0\n"
                              "@@assoc\n"
                              "abc_a100|velatkilic|abc_debug,abc_normal|abc_debug\n"
                              "abc_a100|abc_lab|abc_normal|\n"
                              "dgx|abc_lab|dgx_shared|dgx_shared\n"
                              "cpu|velatkilic||\n"
                              "@@rc 0\n"
                              "@@qos\n"
                              "normal|\n"
                              "abc_debug|01:00:00\n"
                              "abc_normal|2-00:00:00\n"
                              "@@rc 0\n"
                              "@@scontrol\n"
                              "PartitionName=cpu Default=YES OverSubscribe=NO State=UP\n"
                              "PartitionName=dgx Default=NO OverSubscribe=EXCLUSIVE State=UP\n"
                              "@@rc 0\n"
                              "@@end\n";
} // namespace

TEST_CASE("cluster: sinfo, sacctmgr and scontrol are read into partitions", "[app][cluster]") {
    const cluster::ClusterInfo info = cluster::parseClusterInfo(kInfoOutput);
    CHECK(info.error.empty());
    CHECK(info.user == "tester");
    REQUIRE(info.partitions.size() == 4);
    const cluster::Partition* cpu = cluster::findPartition(info, "cpu");
    REQUIRE(cpu);
    CHECK(cpu->isDefault);
    CHECK(cpu->nodes == 10);
    CHECK(cpu->idle == 6);
    CHECK(cpu->gpusPerNode == 0);
    CHECK(cpu->cpusPerNode == 64);
    const cluster::Partition* a100 = cluster::findPartition(info, "abc_a100");
    REQUIRE(a100);
    CHECK_FALSE(a100->isDefault);
    CHECK(a100->nodes == 4);
    CHECK(a100->idle == 2);
    CHECK(a100->mixed == 1);   // the drained node, not responding, is neither
    CHECK(a100->gpusPerNode == 1);
    CHECK(a100->gpuType == "a100");
    CHECK(a100->memPerNodeMB == 500000);
    CHECK(a100->maxTime == "3-00:00:00");
    CHECK(cluster::partitionSummary(*a100) == "4 nodes (2 idle, 1 partly used) \xC2\xB7 1x A100 per node \xC2\xB7 max 3-00:00:00");
    CHECK(cluster::partitionWarning(*a100).empty());
    const cluster::Partition* h100 = cluster::findPartition(info, "lab_h100");
    REQUIRE(h100);
    CHECK(h100->gpusPerNode == 4);   // the shards are not GPUs; the socket list's comma splits nothing
    CHECK(h100->cpusPerNode == 96);
    CHECK(cluster::partitionSummary(*h100).find("none free") != std::string::npos);
    const cluster::Partition* dgx = cluster::findPartition(info, "dgx");
    REQUIRE(dgx);
    CHECK(dgx->exclusive);
    CHECK(dgx->gpusPerNode == 8);
    CHECK(cluster::partitionWarning(*dgx).find("all 8 A100s") != std::string::npos);
    // a whole-node partition by scontrol alone
    cluster::Partition whole = *a100;
    whole.name = "big";
    whole.exclusive = true;
    CHECK(cluster::partitionWarning(whole).find("OverSubscribe=EXCLUSIVE") != std::string::npos);

    // the associations
    CHECK(info.associationsKnown);
    CHECK(info.exclusiveKnown);
    CHECK(cluster::hasAssociation(info, "abc_a100"));
    CHECK(cluster::hasAssociation(info, "cpu"));
    CHECK_FALSE(cluster::hasAssociation(info, "lab_h100"));
    CHECK(cluster::accountsFor(info, "abc_a100") == std::vector<std::string>{"velatkilic", "abc_lab"});
    CHECK(cluster::qosFor(info, "abc_a100", "velatkilic") == std::vector<std::string>{"abc_debug", "abc_normal"});
    CHECK(cluster::qosFor(info, "cpu", "velatkilic").empty());
    CHECK(info.qosMaxWall.at("abc_debug") == "01:00:00");
}

TEST_CASE("cluster: what cannot be asked is unknown, not empty", "[app][cluster]") {
    // sacctmgr refused, scontrol missing: the partitions are all listed as usable
    cluster::ClusterInfo info = cluster::parseClusterInfo("@@user u\n@@sinfo\nabc|up|infinite|1|idle|gpu:2|8|1000\n@@rc 0\n"
                                                          "@@assoc\nsacctmgr: error: Problem talking to the database\n@@rc 1\n"
                                                          "@@qos\nsacctmgr: error: Problem talking to the database\n@@rc 1\n"
                                                          "@@scontrol\nbash: scontrol: command not found\n@@rc 127\n@@end\n");
    CHECK(info.error.empty());
    REQUIRE(info.partitions.size() == 1);
    CHECK(info.partitions[0].gpusPerNode == 2);
    CHECK(info.partitions[0].gpuType.empty());
    CHECK(cluster::partitionSummary(info.partitions[0]) == "1 node (1 idle) \xC2\xB7 2x GPU per node \xC2\xB7 no time limit");
    CHECK_FALSE(info.associationsKnown);
    CHECK_FALSE(info.exclusiveKnown);
    CHECK(cluster::hasAssociation(info, "abc"));
    REQUIRE(info.notes.size() == 1);
    CHECK(info.notes[0].find("Problem talking to the database") != std::string::npos);
    // sinfo itself failed: why, in its own words
    info = cluster::parseClusterInfo("@@user u\n@@sinfo\nslurm_load_partitions: Unable to contact slurm controller\n@@rc 1\n@@end\n");
    CHECK(info.partitions.empty());
    CHECK(info.error.find("Unable to contact slurm controller") != std::string::npos);
    // a cut-off answer
    info = cluster::parseClusterInfo("@@user u\n@@sinfo\nabc|up|1:00:00|1|idle|(null)|8|1000\n");
    CHECK_FALSE(info.error.empty());
    CHECK(cluster::parseClusterInfo("").partitions.empty());
    // the script asks with fixed text only
    const std::string script = cluster::clusterInfoScript();
    CHECK(script.find("sinfo -h -o '%P|%a|%l|%D|%t|%G|%c|%m'") != std::string::npos);
    CHECK(script.find("sacctmgr -n -P show assoc user=\"$u\" format=partition,account,qos,defaultqos") != std::string::npos);
}

TEST_CASE("cluster: Slurm's times and memory sizes", "[app][cluster]") {
    CHECK(cluster::slurmTimeSeconds("90") == 90 * 60);
    CHECK(cluster::slurmTimeSeconds("10:30") == 10 * 60 + 30);
    CHECK(cluster::slurmTimeSeconds("01:00:00") == 3600);
    CHECK(cluster::slurmTimeSeconds("2-12") == 2 * 86400 + 12 * 3600);
    CHECK(cluster::slurmTimeSeconds("2-12:30") == 2 * 86400 + 12 * 3600 + 30 * 60);
    CHECK(cluster::slurmTimeSeconds("3-00:00:00") == 3 * 86400);
    CHECK(cluster::slurmTimeSeconds("infinite") == -1);
    CHECK(cluster::slurmTimeSeconds("UNLIMITED") == -1);
    CHECK(cluster::slurmTimeSeconds("") == -2);
    CHECK(cluster::slurmTimeSeconds("1:x:00") == -2);
    CHECK(cluster::slurmTimeSeconds("1:2:3:4") == -2);
    CHECK(cluster::slurmTimeText(3600) == "01:00:00");
    CHECK(cluster::slurmTimeText(3 * 86400 + 61) == "3-00:01:01");
    CHECK(cluster::memoryMB("64G") == 65536);
    CHECK(cluster::memoryMB("64GB") == 65536);
    CHECK(cluster::memoryMB("500M") == 500);
    CHECK(cluster::memoryMB("1T") == 1024 * 1024);
    CHECK(cluster::memoryMB("4096") == 4096);
    CHECK(cluster::memoryMB("lots") == -1);
    CHECK(cluster::memoryMB("") == -1);
}

TEST_CASE("cluster: choosing a partition fills the account and QoS and keeps within its nodes", "[app][cluster]") {
    const cluster::ClusterInfo info = cluster::parseClusterInfo(kInfoOutput);
    cluster::Profile p;
    p.account = "someone_else";
    p.qos = "high";
    p.time = "2-00:00:00";
    p.gpus = 4;
    p.cpus = 64;
    p.mem = "1T";
    std::vector<std::string> changed = cluster::choosePartition(p, info, "abc_a100");
    CHECK(p.partition == "abc_a100");
    CHECK(p.account == "velatkilic");   // the partition's first association
    CHECK(p.qos == "abc_debug");        // its default QoS
    CHECK(p.time == "01:00:00");        // abc_debug's MaxWall, below the partition's 3 days
    CHECK(p.gpus == 1);
    CHECK(p.cpus == 32);
    CHECK(p.mem == "488G");             // a node's 500000 MB
    CHECK(changed.size() == 6);

    // the profile's own account and QoS stay when the association has them
    p.account = "abc_lab";
    p.qos = "abc_normal";
    p.time = "12:00:00";
    changed = cluster::choosePartition(p, info, "abc_a100");
    CHECK(p.account == "abc_lab");
    CHECK(p.qos == "abc_normal");
    CHECK(p.time == "12:00:00");
    CHECK(changed.empty());

    // an account without that partition: the one that has it
    p.account = "velatkilic";
    changed = cluster::choosePartition(p, info, "dgx");
    CHECK(p.account == "abc_lab");
    CHECK(p.qos == "dgx_shared");
    CHECK(p.time == "12:00:00");   // dgx_shared has no MaxWall here; the partition allows a day

    // an association with no QoS of its own: none is asked for
    p.qos = "abc_debug";
    cluster::choosePartition(p, info, "cpu");
    CHECK(p.account == "velatkilic");
    CHECK(p.qos.empty());

    // no association: the account and QoS are the user's to type; the nodes still bound the rest
    p.account = "mine";
    p.qos = "q";
    p.gpus = 8;
    cluster::choosePartition(p, info, "lab_h100");
    CHECK(p.account == "mine");
    CHECK(p.qos == "q");
    CHECK(p.gpus == 4);

    // a partition sinfo does not list (typed by hand): only the name changes
    p.gpus = 8;
    CHECK(cluster::choosePartition(p, info, "elsewhere").empty());
    CHECK(p.partition == "elsewhere");
    CHECK(p.gpus == 8);
}

TEST_CASE("cluster: the profile remembers the Slurm choice of each host", "[app][cluster]") {
    cluster::Profile p;
    p.host = "fiona";
    p.partition = "dgx";
    p.account = "abc_lab";
    p.qos = "dgx_shared";
    p.time = "02:00:00";
    p.remember();
    p.host = "other";
    p.partition = "gpu";
    p.account = "me";
    p.qos = "";
    p.time = "00:30:00";
    p.remember();
    const cluster::Profile back = cluster::Profile::fromJson(p.toJson());
    REQUIRE(back.perHost.size() == 2);
    cluster::Profile q = back;
    CHECK(q.recall("fiona"));
    CHECK(q.partition == "dgx");
    CHECK(q.account == "abc_lab");
    CHECK(q.qos == "dgx_shared");
    CHECK(q.time == "02:00:00");
    CHECK_FALSE(q.recall("nowhere"));
    CHECK(q.partition == "dgx");
    CHECK(q.recall("other"));
    CHECK(q.partition == "gpu");
    CHECK(q.qos.empty());
    // an old profile without the map reads as before
    const cluster::Profile old = cluster::Profile::fromJson(nlohmann::json{{"host", "fiona"}, {"partition", "abc_a100"}});
    CHECK(old.perHost.empty());
    CHECK(old.partition == "abc_a100");
}

TEST_CASE("cluster: log in lists the partitions and submits nothing", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    cluster::Session session;
    session.setAskpassProgram(SIRIUS_TEST_ASKPASS);
    cluster::Profile p;
    p.host = "fakecluster";
    p.sshProgram = fc.python;
    p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
    CHECK_FALSE(session.clusterInfo());
    session.logIn(p);
    REQUIRE(waitFor([&] { return session.status().state != cluster::State::Connecting; }, std::chrono::seconds(60)));
    cluster::Status st = session.status();
    INFO(st.reason);
    CHECK(st.state == cluster::State::Idle);
    CHECK(st.sshUp);
    CHECK(st.steps[0].status == cluster::StepStatus::Done);
    for (int i = 1; i < cluster::kStepCount; ++i) CHECK(st.steps[static_cast<std::size_t>(i)].status == cluster::StepStatus::Pending);
    std::optional<cluster::ClusterInfo> info = session.clusterInfo();
    REQUIRE(info);
    INFO(info->error);
    CHECK(info->host == "fakecluster");
    CHECK(info->user == "tester");
    CHECK(info->error.empty());
    REQUIRE(info->partitions.size() == 4);
    CHECK(info->associationsKnown);
    CHECK(info->exclusiveKnown);
    const cluster::Partition* dgx = cluster::findPartition(*info, "dgx");
    REQUIRE(dgx);
    CHECK(dgx->exclusive);
    CHECK_FALSE(cluster::hasAssociation(*info, "lab_h100"));
    // the user's name was the cluster's own
    CHECK(readAll(fc.slurm / "sacctmgr.args").find("user=tester ") != std::string::npos);
    // nothing was submitted
    CHECK_FALSE(fs::exists(fc.slurm / "next"));
    CHECK_FALSE(session.hasJob());

    // Refresh, with the accounting refused: the partitions stay, the associations are unknown
    std::ofstream(fc.slurm / "sacctmgr.fail") << "1";   // the session is open: its environment is set
    const cluster::ClusterInfo again = session.refreshClusterInfo();
    CHECK(again.partitions.size() == 4);
    CHECK_FALSE(again.associationsKnown);
    CHECK(cluster::hasAssociation(again, "lab_h100"));
    CHECK_FALSE(again.notes.empty());
    CHECK_FALSE(session.queryingClusterInfo());
    // sinfo refused: said why
    std::ofstream(fc.slurm / "sinfo.fail") << "1";
    CHECK(session.refreshClusterInfo().error.find("Unable to contact slurm controller") != std::string::npos);
    session.disconnect(false);
    // the last answer is kept after the disconnect
    REQUIRE(session.clusterInfo());
    CHECK_THROWS_AS(session.refreshClusterInfo(), ssh::SshError);
}

// --- the worker in a container image ------------------------------------------------------

namespace {
    // A checkout on the "cluster": the worker package and the job template.
    void copyCheckout(const fs::path& checkout) {
        fs::create_directories(checkout / "app" / "python");
        for (const char* d : {"sirius_worker", "slurm"})
            fs::copy(fs::path(SIRIUS_TEST_SOURCE_DIR) / "app" / "python" / d, checkout / "app" / "python" / d,
                     fs::copy_options::recursive | fs::copy_options::skip_existing);
    }

    fs::path containerSite(const FakeCluster& fc) { return imageSite(fc.root); }

    cluster::Profile containerProfile(const FakeCluster& fc, const std::string& image) {
        cluster::Profile p;
        p.host = "fakecluster";
        p.sshProgram = fc.python;
        p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
        p.checkout = "~/sirius";
        p.container = image;
        p.engine = false;
        p.gpus = 0;
        p.port = ssh::freeLocalPort();
        return p;
    }

    cluster::Status connectUntilSettled(cluster::Session& session, const cluster::Profile& p) {
        session.connect(p);
        waitFor([&] { return settled(session); }, std::chrono::seconds(180));
        return session.status();
    }

    // The worker (again) in the job the session holds.
    cluster::Status workerUntilSettled(cluster::Session& session, const cluster::Profile& p) {
        session.startWorker(p);
        waitFor([&] { return session.status().state != cluster::State::JobReady && session.status().state != cluster::State::Connected; },
                std::chrono::seconds(5));
        waitFor([&] { return settled(session); }, std::chrono::seconds(180));
        return session.status();
    }

    // No worker step was started (srun.args names none; the checks before
    // a start are a step of their own, sirius-check).
    bool noWorkerStarted(const FakeCluster& fc) { return readAll(fc.slurm / "srun.args").find("--job-name=sirius-worker") == std::string::npos; }
} // namespace

TEST_CASE("cluster: the profile keeps the container image and its launcher", "[app][cluster]") {
    cluster::Profile p;
    CHECK(p.container.empty());
    CHECK(p.launcher == "apptainer");
    p.container = "/clusterfs/nvme2/Users/u/sirius-worker.sif";
    p.launcher = "singularity";
    const cluster::Profile back = cluster::Profile::fromJson(p.toJson());
    CHECK(back.container == p.container);
    CHECK(back.launcher == "singularity");
    CHECK(cluster::Profile::fromJson(nlohmann::json{{"launcher", ""}}).launcher == "apptainer");
}

TEST_CASE("cluster: the profile keeps the container's binds and Python path, per host too", "[app][cluster]") {
    cluster::Profile p;
    CHECK(p.bind.empty());
    CHECK(p.containerPythonPath.empty());
    p.host = "fiona";
    p.container = "~/sirius-worker.sif";
    p.bind = "/clusterfs:/clusterfs,/global/scratch";
    p.containerPythonPath = "/opt/extra:/opt/more";
    p.remember();
    p.host = "other";
    p.bind = "/data";
    p.containerPythonPath.clear();
    p.remember();
    const cluster::Profile back = cluster::Profile::fromJson(p.toJson());
    CHECK(back.bind == "/data");
    CHECK(back.containerPythonPath.empty());
    cluster::Profile q = back;
    REQUIRE(q.recall("fiona"));
    CHECK(q.bind == "/clusterfs:/clusterfs,/global/scratch");
    CHECK(q.containerPythonPath == "/opt/extra:/opt/more");
    REQUIRE(q.recall("other"));
    CHECK(q.bind == "/data");
    CHECK(q.containerPythonPath.empty());
    // a host remembered before binds were kept leaves the profile's as they are
    nlohmann::json old = q.toJson();
    old.erase("last_used");
    old["perHost"] = {{"fiona", {{"partition", "dgx"}, {"account", "a"}, {"qos", "q"}, {"time", "01:00:00"}}}};
    cluster::Profile r = cluster::Profile::fromJson(old);
    REQUIRE(r.recall("fiona"));
    CHECK(r.partition == "dgx");
    CHECK(r.bind == "/data");
    // an old profile without the fields reads as empty
    const cluster::Profile none = cluster::Profile::fromJson(nlohmann::json{{"host", "fiona"}});
    CHECK(none.bind.empty());
    CHECK(none.containerPythonPath.empty());
}

TEST_CASE("cluster: the bind list's host paths, the empty-bind warning and paths the container cannot see", "[app][cluster]") {
    using V = std::vector<std::string>;
    CHECK(cluster::bindHostPaths("") == V{});
    CHECK(cluster::bindHostPaths("/clusterfs:/clusterfs, /global/scratch:/scratch:ro,,  ") == V{"/clusterfs", "/global/scratch"});
    CHECK(cluster::bindHostPaths("~/data") == V{"~/data"});

    cluster::Profile p;
    p.checkout = "~/dev/sirius";
    CHECK(cluster::emptyBindWarning(p).empty());   // no container: nothing to bind
    p.container = "~/w.sif";
    const std::string w = cluster::emptyBindWarning(p);
    CHECK(w.find("only itself and your home folder") != std::string::npos);
    CHECK(w.find("add those folders under Data folders") != std::string::npos);
    p.bind = "/clusterfs";
    CHECK(cluster::emptyBindWarning(p).empty());

    const std::string home = "/global/home/users/u";
    // inside a bind, the home folder, the checkout or /tmp: seen
    CHECK(cluster::unboundPathMessage(p, home, "/clusterfs/nvme/a.tif").empty());
    CHECK(cluster::unboundPathMessage(p, home, "/clusterfs").empty());
    CHECK(cluster::unboundPathMessage(p, home, home + "/data/a.tif").empty());
    CHECK(cluster::unboundPathMessage(p, home, "~/data/a.tif").empty());
    CHECK(cluster::unboundPathMessage(p, home, "/tmp/a.tif").empty());
    // elsewhere: said, with the top folder to add
    std::string m = cluster::unboundPathMessage(p, home, "/global/scratch/u/a.tif");
    CHECK(m.find("not bound into the worker's container") != std::string::npos);
    CHECK(m.find("add its folder under Data folders") != std::string::npos);
    CHECK(m.find("/global") != std::string::npos);
    // a prefix of a name is not a parent folder
    CHECK_FALSE(cluster::unboundPathMessage(p, home, "/clusterfs2/a.tif").empty());
    // binds with a destination and options, and a "~/" bind
    p.bind = "/global/scratch:/scratch:ro,~/elsewhere";
    CHECK(cluster::unboundPathMessage(p, home, "/global/scratch/u/a.tif").empty());
    CHECK_FALSE(cluster::unboundPathMessage(p, home, "/clusterfs/a.tif").empty());
    // a checkout outside the home folder is bound by the job
    p.checkout = "/opt/sirius";
    CHECK(cluster::unboundPathMessage(p, home, "/opt/sirius/data/a.tif").empty());
    // nothing to say without a container or before $HOME is known
    CHECK(cluster::unboundPathMessage(p, "", "/clusterfs/a.tif").empty());
    p.container.clear();
    CHECK(cluster::unboundPathMessage(p, home, "/clusterfs/a.tif").empty());
}

TEST_CASE("cluster: the checks of a container image say what is wrong and never suggest pip", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "broken.sif") << "broken\n";
    std::ofstream(fc.home / "empty.sif") << "an image without sirius\n";
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    // a failed check leaves the job held, for the next try in it
    auto checksFailed = [](const cluster::Status& st) {
        return st.state == cluster::State::JobReady && st.steps[static_cast<int>(cluster::Step::Checks)].status == cluster::StepStatus::Failed;
    };

    // no image at all: said, with where one comes from
    cluster::Status st = connectUntilSettled(session, containerProfile(fc, ""));
    INFO(st.reason);
    CHECK(checksFailed(st));
    CHECK(st.reason.find("No worker image is set") != std::string::npos);
    CHECK(st.fix.find("Worker image") != std::string::npos);
    CHECK(st.jobId == "4711");   // the job came first, and stays

    // no image there
    st = workerUntilSettled(session, containerProfile(fc, "~/missing.sif"));
    INFO(st.reason);
    CHECK(checksFailed(st));
    CHECK(st.reason.find("no worker image at ~/missing.sif") != std::string::npos);
    CHECK(st.fix.find("Build the worker image") != std::string::npos);
    CHECK(st.fix.find("pip") == std::string::npos);

    // an image that does not open
    st = workerUntilSettled(session, containerProfile(fc, "~/broken.sif"));
    CHECK(checksFailed(st));
    CHECK(st.reason.find("cannot run the worker") != std::string::npos);
    CHECK(st.remoteOutput.find("squashfs") != std::string::npos);
    CHECK(st.fix.find("pip") == std::string::npos);

    // an image without the sirius package (the checkout's folder named sirius does not count),
    // with numpy, so that it is sirius that is missing whatever the interpreter behind the fake has
    const fs::path numpyOnly = fc.root / "numpy-only-site";
    fs::create_directories(numpyOnly / "numpy");
    std::ofstream(numpyOnly / "numpy" / "__init__.py") << "__version__ = '0-test'\n";
    // through a file: the fake ssh session is already open, so a new environment
    // variable would not reach the fake apptainer
    std::ofstream(fc.slurm / "container_site") << numpyOnly.string();
    st = workerUntilSettled(session, containerProfile(fc, "~/empty.sif"));
    fs::remove(fc.slurm / "container_site");
    CHECK(checksFailed(st));
    CHECK(st.reason.find("sirius and numpy do not import") != std::string::npos);
    // Where a checkout folder named sirius is importable, Python finds an empty
    // namespace package and the check's own assertion names it; elsewhere it is
    // plain "No module named 'sirius'". Both are the missing compiled package.
    INFO(st.remoteOutput);
    CHECK((st.remoteOutput.find("not the compiled package") != std::string::npos ||
           st.remoteOutput.find("No module named 'sirius'") != std::string::npos));

    // neither apptainer nor singularity
    fs::remove(fc.bin / "apptainer");
    st = workerUntilSettled(session, containerProfile(fc, "~/empty.sif"));
    CHECK(checksFailed(st));
    CHECK(st.reason.find("Neither apptainer nor singularity") != std::string::npos);
    CHECK(noWorkerStarted(fc));         // the checks stopped every try before the worker
    CHECK_FALSE(fs::exists(fc.slurm / "4712.args"));   // all in the one job
    session.disconnect(true);
}

TEST_CASE("cluster: the job template runs the worker in the container", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "w.sif") << "image\n";
    // only singularity on this "cluster": apptainer, the default, is not found
    fs::rename(fc.bin / "apptainer", fc.bin / "singularity");
    ssh::Session s;
    s.open(fc.options(), {}, std::chrono::seconds(60));
    // the real sirius_worker.sbatch, run by bash as Slurm would (the fake launcher runs nothing)
    const std::string job = "cd ~/sirius && mkdir -p ~/.sirius/run && export FAKE_APPTAINER_DRY=1 SIRIUS_CONTAINER=\"$HOME/w.sif\" "
                            "SIRIUS_TOKEN_FILE=\"$HOME/.sirius/run/token.x\" SLURM_SUBMIT_DIR=\"$PWD\" && ";
    ssh::CommandResult r = s.run(job + "bash app/python/slurm/sirius_worker.sbatch");
    INFO(r.out << "\n"
               << r.err);
    CHECK(r.ok());
    std::string args = readAll(fc.slurm / "apptainer.args");
    // the environment (and so the token's file name) goes in through a private --env-file, never argv
    CHECK(args.rfind("singularity exec --nv --cleanenv ", 0) == 0);
    CHECK(args.find("--env-file ") != std::string::npos);
    CHECK(args.find("SIRIUS_TOKEN") == std::string::npos);
    CHECK(args.find("w.sif python -m sirius_worker --host 0.0.0.0 --port 0 --device cuda --max-clients 8") != std::string::npos);
    // a CPU job gets no --nv
    fs::remove(fc.slurm / "apptainer.args");
    r = s.run(job + "SIRIUS_DEVICE=cpu bash app/python/slurm/sirius_worker.sbatch");
    CHECK(r.ok());
    args = readAll(fc.slurm / "apptainer.args");
    CHECK(args.find("--nv") == std::string::npos);
    CHECK(args.find("--device cpu") != std::string::npos);
    // no launcher at all: said so, nothing run
    fs::remove(fc.bin / "singularity");
    r = s.run(job + "bash app/python/slurm/sirius_worker.sbatch");
    CHECK_FALSE(r.ok());
    CHECK(r.err.find("neither apptainer nor singularity") != std::string::npos);  // the message names both
    s.close();
}

TEST_CASE("cluster: connect runs the worker in the container image", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the worker");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "sirius-worker.sif") << "image\n";
    setEnv("FAKE_CONTAINER_SITE", containerSite(fc).string());
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    cluster::Profile p = containerProfile(fc, "~/sirius-worker.sif");
    p.gpus = 0;
    const cluster::Status st = connectUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    const std::string checks = st.steps[static_cast<int>(cluster::Step::Checks)].detail;
    CHECK(checks.find("apptainer") != std::string::npos);
    CHECK(checks.find("sirius, numpy") != std::string::npos);
    // the worker step was told the image and the launcher
    const std::string env = readAll(fc.slurm / "4711.env");
    CHECK(env.find("SIRIUS_CONTAINER=") != std::string::npos);
    CHECK(env.find("sirius-worker.sif") != std::string::npos);
    CHECK(env.find("SIRIUS_LAUNCHER=apptainer") != std::string::npos);
    CHECK(env.find("SIRIUS_VENV") == std::string::npos);
    CHECK(env.find("SIRIUS_DEVICE=cpu") != std::string::npos);
    // the worker ran through the launcher, the token file read from the bound ~/.sirius/run
    const std::string args = readAll(fc.slurm / "apptainer.args");
    CHECK(args.find("--bind") != std::string::npos);
    CHECK(args.find("python -m sirius_worker") != std::string::npos);
    CHECK(args.find("--nv") == std::string::npos);
    for (const auto& entry : fs::directory_iterator(fc.home / ".sirius" / "run")) CHECK(entry.path().filename().string().rfind("token.", 0) != 0);
    session.disconnect(true);
    setEnv("FAKE_CONTAINER_SITE", "");
}

TEST_CASE("cluster: the checks fail on a data folder that is not there, before the worker starts", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "w.sif") << "image\n";
    fs::create_directories(fc.home / "data");
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    cluster::Profile p = containerProfile(fc, "~/w.sif");
    p.bind = "~/data:/data,~/missing:/m";
    const cluster::Status st = connectUntilSettled(session, p);
    INFO(st.reason);
    CHECK(st.state == cluster::State::JobReady);   // the job holds on
    CHECK(st.steps[static_cast<int>(cluster::Step::Checks)].status == cluster::StepStatus::Failed);
    CHECK(st.reason.find("~/missing") != std::string::npos);
    CHECK(st.reason.find("~/data") == std::string::npos);   // the one that is there is not named
    CHECK(st.fix.find("Data folders") != std::string::npos);
    CHECK(noWorkerStarted(fc));
    session.disconnect(true);
}

TEST_CASE("cluster: no data folder is a warning; the data folders and the Python path reach the worker", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the image check");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "w.sif") << "image\n";
    fs::create_directories(fc.home / "data");
    setEnv("FAKE_CONTAINER_SITE", containerSite(fc).string());
    // srun refuses every worker step once it has recorded its environment: no worker is started
    std::ofstream(fc.slurm / "srun.fail") << "1\n";
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    cluster::Profile p = containerProfile(fc, "~/w.sif");

    // no data folder: the checks pass with a warning, and the worker is started all the same
    cluster::Status st = connectUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    const cluster::StepState checks = st.steps[static_cast<int>(cluster::Step::Checks)];
    CHECK(checks.status == cluster::StepStatus::Warning);
    CHECK(checks.detail.find("only itself and your home folder") != std::string::npos);
    CHECK(checks.detail.find("add those folders under Data folders") != std::string::npos);
    CHECK(st.steps[static_cast<int>(cluster::Step::Start)].status == cluster::StepStatus::Failed);
    CHECK(st.state == cluster::State::JobReady);
    CHECK_FALSE(st.home.empty());   // the checks said where $HOME is
    std::string env = readAll(fc.slurm / "4711.env");
    CHECK(env.find("SIRIUS_CONTAINER=") != std::string::npos);
    CHECK(env.find("SIRIUS_CONTAINER_BIND") == std::string::npos);
    CHECK(env.find("SIRIUS_CONTAINER_PYTHONPATH") == std::string::npos);

    // binds ("~/" made $HOME) and a Python path: checked, then in the job's environment
    fs::remove(fc.slurm / "apptainer.args");
    // (the fake apptainer mounts nothing: a folder is checked "inside the image" at its own path)
    p.bind = "~/data, /tmp";
    p.containerPythonPath = "/opt/extra";
    st = workerUntilSettled(session, p);
    const cluster::StepState checks2 = st.steps[static_cast<int>(cluster::Step::Checks)];
    INFO(checks2.detail);
    CHECK(checks2.status == cluster::StepStatus::Done);
    CHECK(checks2.detail.find("2 data folders") != std::string::npos);
    env = readAll(fc.slurm / "4711.env");   // the same job's next step
    INFO(env);
    CHECK(env.find("SIRIUS_CONTAINER_BIND=") != std::string::npos);
    CHECK(env.find("/data,/tmp\n") != std::string::npos);
    CHECK(env.find("SIRIUS_CONTAINER_BIND=~") == std::string::npos);
    CHECK(env.find("SIRIUS_CONTAINER_PYTHONPATH=/opt/extra\n") != std::string::npos);
    CHECK(env.find("SIRIUS_TOKEN=") == std::string::npos);   // never the token itself
    // the image was tried with the same binds
    CHECK(readAll(fc.slurm / "apptainer.args").find("--bind ") != std::string::npos);
    CHECK(readAll(fc.slurm / "apptainer.args").find("/data,/tmp") != std::string::npos);
    // and each folder checked in the job, on its node, inside the image
    bool dataRow = false;
    for (const cluster::NodeCheck& c : st.nodeChecks)
        if (c.name == "data:0") dataRow = c.status == cluster::StepStatus::Done && c.detail == "~/data";
    CHECK(dataRow);
    session.disconnect(true);
    setEnv("FAKE_CONTAINER_SITE", "");
}

// --- SIRIUS's engine as the job (the HPC engine plan, P2) ---------------------------------

namespace {
    // A uint16 ImageJ stack (t, z, y, x) in the cluster's home: a blob over a ramp.
    void writeStackTiff(const fs::path& path, sirius::Index t, sirius::Index z, sirius::Index y, sirius::Index x) {
        sirius::Buffer<std::uint16_t> stack(sirius::Shape{t * z, y, x});
        for (sirius::Index i = 0; i < t * z; ++i)
            for (sirius::Index r = 0; r < y; ++r)
                for (sirius::Index c = 0; c < x; ++c) {
                    const double dx = static_cast<double>(c) - 20.0, dy = static_cast<double>(r) - 16.0;
                    stack.data()[(i * y + r) * x + c] =
                        static_cast<std::uint16_t>(100.0 + 1500.0 * std::exp(-(dx * dx + dy * dy) / 40.0) + static_cast<double>((r + c + i) % 13));
                }
        sirius::TiffWriteOptions o;
        o.description = "ImageJ=1.53t\nimages=" + std::to_string(t * z) + "\nchannels=1\nslices=" + std::to_string(z) + "\nframes=" + std::to_string(t) +
                        "\nhyperstack=true\nspacing=0.3\nunit=micron\n";
        o.xPixelUm = 0.1;
        o.yPixelUm = 0.1;
        sirius::writeTiffStack<std::uint16_t>(path.string(), stack.view(), o);
    }

    cluster::Profile engineProfile(const FakeCluster& fc) {
        std::ofstream(fc.home / "w.sif") << "image\n";
        setEnv("FAKE_CONTAINER_SITE", imageSite(fc.root).string());
        cluster::Profile p;
        p.host = "fakecluster";
        p.sshProgram = fc.python;
        p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
        p.checkout = "~/sirius";
        p.container = "~/w.sif";
        p.port = ssh::freeLocalPort();
        p.engine = true;
        p.engineBin = SIRIUS_TEST_CLI;
        p.gpus = 0;
        return p;
    }

    struct ScratchDir {
        fs::path path = sirius::test::uniqueTempPath("cluster-wb", "");
        ~ScratchDir() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    };

    // A detached fake job (FAKE_SLURM_DETACH) outlives the ssh session: one the
    // test did not cancel is stopped here, whatever the test's outcome.
    struct JobReaper {
        fs::path slurm;
        ~JobReaper() {
            std::error_code ec;
            for (const auto& e : fs::directory_iterator(slurm, ec)) {
                const std::string ext = e.path().extension().string();
                if (ext != ".winpid" && ext != ".ospid") continue;
                const std::string pid = readAll(e.path()).substr(0, readAll(e.path()).find_first_of("\r\n"));
                if (pid.empty()) continue;
                ChildProcess k;
                ChildProcess::Options o;
#ifdef _WIN32
                o.program = "taskkill";
                o.arguments = {"/F", "/T", "/PID", pid};
#else
                o.program = "kill";
                o.arguments = {pid};
#endif
                if (k.start(o)) k.waitForExit(10000);
            }
        }
    };
} // namespace

TEST_CASE("cluster: the profile keeps the engine, on by default", "[app][cluster]") {
    cluster::Profile p;
    CHECK(p.engine);
    p.engine = false;
    p.engineBin = "/opt/sirius/bin/sirius-cli";
    p.engineBuilds = "/shared/sirius-engines";
    const cluster::Profile back = cluster::Profile::fromJson(p.toJson());
    CHECK_FALSE(back.engine);
    CHECK(back.engineBin == "/opt/sirius/bin/sirius-cli");
    CHECK(back.engineBuilds == "/shared/sirius-engines");
    CHECK(cluster::Profile::fromJson(nlohmann::json{{"container", "~/w.sif"}}).engine);
    CHECK(cluster::Profile::fromJson(nlohmann::json{{"container", ""}}).engine);
    CHECK_FALSE(cluster::Profile::fromJson(nlohmann::json{{"container", "~/w.sif"}, {"engine", false}}).engine);
}

TEST_CASE("cluster: the job template runs SIRIUS's engine, in the container or not", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "w.sif") << "image\n";
    ssh::Session s;
    s.open(fc.options(), {}, std::chrono::seconds(60));
    const std::string job = "cd ~/sirius && mkdir -p ~/.sirius/run && export FAKE_APPTAINER_DRY=1 SIRIUS_CONTAINER=\"$HOME/w.sif\" SIRIUS_ENGINE=1 "
                            "SIRIUS_TOKEN_FILE=\"$HOME/.sirius/run/token.x\" SLURM_SUBMIT_DIR=\"$PWD\" && ";
    ssh::CommandResult r = s.run(job + "bash app/python/slurm/sirius_worker.sbatch");
    INFO(r.out << "\n"
               << r.err);
    CHECK(r.ok());
    const std::string args = readAll(fc.slurm / "apptainer.args");
    INFO(args);
    // the image's engine, its Python worker beside it from the checkout
    CHECK(args.find("w.sif /opt/sirius/bin/sirius-cli serve --host 0.0.0.0 --port 0 --device cuda --max-clients 8 --python python --worker-dir ") !=
          std::string::npos);
    CHECK(args.find("SIRIUS_TOKEN") == std::string::npos);
    CHECK(r.out.find("SIRIUS engine: /opt/sirius/bin/sirius-cli serve") != std::string::npos);
    s.close();
}

TEST_CASE("cluster: connect starts SIRIUS's engine; a pipeline runs there and stays there; a reconnect reattaches; the job's end takes it",
          "[app][cluster][engine]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the checks");
    copyCheckout(fc.home / "sirius");
    writeStackTiff(fc.home / "raw.tif", 2, 6, 32, 40);
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    // a disconnect leaves the job running here, as on a cluster: the reconnect finds it
    setEnv("FAKE_SLURM_KILL_ON_EXIT", "0");
    setEnv("FAKE_SLURM_DETACH", "1");
    JobReaper reaper{fc.slurm};
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    const cluster::Profile p = engineProfile(fc);
    cluster::Status st = connectUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    REQUIRE(st.caps.engine.is_object());
    CHECK(st.caps.engine["build"] == buildInfo().build);
    CHECK(st.steps[static_cast<int>(cluster::Step::Hello)].detail.find("SIRIUS engine") != std::string::npos);
    const std::string env = readAll(fc.slurm / "4711.env");
    CHECK(env.find("SIRIUS_ENGINE=1") != std::string::npos);
    CHECK(env.find("SIRIUS_ENGINE_BIN=") != std::string::npos);
    const std::string session1 = st.caps.engine.value("session", std::string());
    CHECK_FALSE(session1.empty());

    // the application: the cluster's datasets through the engine, the HPC backend on it
    auto datasets = std::make_shared<RemoteDatasets>("fakecluster", [&] { return session.connectWorker(); });
    datasets->install();
    const cluster::Listing home = session.list("~");
    ScratchDir scratch;
    Workbench wb(scratch.path / "wb");
    wb.openDataset(makeClusterPath("fakecluster", home.path + "/raw.tif"));
    REQUIRE(wb.hasDataset());
    RemoteConfig rc;
    rc.connect = [&session](const std::function<bool()>& cancelled) { return session.connectWorker(std::chrono::seconds(10), cancelled); };
    rc.known = true;
    rc.engine = st.caps.engine;
    rc.where = "fakecluster \xC2\xB7 " + st.node + " \xC2\xB7 job " + st.jobId;
    wb.setRemoteConfig(rc);
    wb.setBackend(Backend::Hpc);
    wb.setHpcDevice(HpcDevice::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("decon", -1, false);
    wb.setStepParam(1, "iterations", std::int64_t{2});
    wb.setStepParam(1, "psf_size", std::int64_t{7});
    wb.addStep("contrast", -1, false);
    wb.setStepCache(2, CachePolicy::Memory);
    const std::uint64_t volumes = RemoteDownloads::volumeBytes(), planes = RemoteDownloads::planeBytes();
    std::shared_ptr<RunJob> job = wb.createRun();
    REQUIRE(job);
    job->execute();
    wb.finishRun(job);
    INFO(job->error());
    REQUIRE(job->succeeded());
    CHECK(job->ranOnEngine());
    // only diagnostics came back: no volume, no plane
    CHECK(RemoteDownloads::volumeBytes() == volumes);
    CHECK(RemoteDownloads::planeBytes() == planes);
    std::shared_ptr<const StepOutput> out = wb.output(2);
    REQUIRE(out);
    auto* node = dynamic_cast<NodeOutputSource*>(const_cast<ArraySource*>(out->source.get()));
    REQUIRE(node);
    CHECK(node->session() == session1);
    CHECK(wb.placementOf(2) == "node CPU");
    ViewRequest req;
    req.index = 3;
    req.factor = 2;
    bool exact = false;
    node->view(req, exact);
    node->waitIdle();
    REQUIRE(node->view(req, exact));
    CHECK(exact);

    // disconnect, leaving the job: connect again reattaches to it, no new job
    session.disconnect(false);
    REQUIRE(session.status().state == cluster::State::Disconnected);
    CHECK_FALSE(session.status().dropped);   // asked for: not a drop
    st = connectUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    CHECK(st.jobId == "4711");
    CHECK_FALSE(fs::exists(fc.slurm / "4712.args"));
    CHECK(st.steps[static_cast<int>(cluster::Step::Submit)].detail.find("reattached") != std::string::npos);
    CHECK(st.steps[static_cast<int>(cluster::Step::Checks)].detail.find("still runs") != std::string::npos);   // its worker too
    CHECK(countOf(readAll(fc.slurm / "srun.args"), "--job-name=sirius-worker") == 1);   // no second worker
    CHECK(st.caps.engine.value("session", std::string()) == session1);
    // the engine kept what it computed: the same handle draws again, through the new tunnel
    ViewRequest other = req;
    other.index = 4;
    node->view(other, exact);
    node->waitIdle();
    INFO(node->lastError());
    REQUIRE(node->view(other, exact));
    CHECK(exact);
    CHECK(wb.outputFresh(2));

    // the job ends (cancelled): its results are gone, said so
    session.disconnect(true);
    CHECK(readAll(fc.slurm / "cancelled").find("4711") != std::string::npos);
    CHECK(wb.nodeOutputsGone(session1, "held by job 4711, which ended (CANCELLED)") == 2);
    CHECK_FALSE(wb.outputFresh(2));
    CHECK(wb.output(2)->gone.find("CANCELLED") != std::string::npos);
    datasets->uninstall();
    setEnv("FAKE_SLURM_KILL_ON_EXIT", "1");
    setEnv("FAKE_SLURM_DETACH", "0");
    setEnv("FAKE_CONTAINER_SITE", "");
}

TEST_CASE("cluster: an engine of other operations is refused at its hello, in words", "[app][cluster][engine]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the checks");
    copyCheckout(fc.home / "sirius");
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    setEnv("SIRIUS_TEST_ENGINE_BUILD", R"({"build": "0.0.9+gdeadbee", "ops_schema": "0000"})");
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    const cluster::Status st = connectUntilSettled(session, engineProfile(fc));
    setEnv("SIRIUS_TEST_ENGINE_BUILD", "");
    setEnv("FAKE_CONTAINER_SITE", "");
    INFO(st.reason);
    CHECK(st.state == cluster::State::JobReady);   // the job holds on for another engine
    CHECK(st.steps[static_cast<int>(cluster::Step::Hello)].status == cluster::StepStatus::Failed);
    CHECK(st.reason.find("0.0.9+gdeadbee") != std::string::npos);
    CHECK(st.reason.find("their operations differ") != std::string::npos);
    session.disconnect(true);
}

// --- the profiles in the settings file ----------------------------------------------------

TEST_CASE("cluster: a new profile has nothing of any site in it", "[app][cluster]") {
    const cluster::Profile p;
    CHECK(p.host.empty());
    CHECK(p.checkout.empty());
    CHECK(p.container.empty());
    CHECK(p.bind.empty());
    CHECK(p.partition.empty());
    CHECK(p.account.empty());
    CHECK(p.qos.empty());
    CHECK(p.engineBuilds.empty());
    CHECK(p.scratch.empty());
    CHECK(p.choices.empty());
    CHECK(p.images.empty());
    CHECK(p.launcher == "apptainer");
    CHECK(p.engine);
    const std::string all = p.toJson().dump();
    for (const char* site : {"fiona", "abc_", "velatkilic", "co_abc", "clusterfs", "venvs", "dev/sirius"}) {
        INFO(site);
        CHECK(all.find(site) == std::string::npos);
    }
    // no settings, no profile: nobody's cluster is assumed
    CHECK(cluster::ProfileBook::fromSettings(nlohmann::json::object()).profiles.empty());
    CHECK(cluster::ProfileBook::fromSettings(nlohmann::json::object()).currentProfile().host.empty());
}

TEST_CASE("cluster: the profile of before becomes [cluster.<host>] with every value; several profiles go through the settings file and back",
          "[app][cluster]") {
    const nlohmann::json legacy = {
        {"host", "login.example.org"}, {"checkout", "~/src/sirius"}, {"venv", "~/v"}, {"container", "/img/w.sif"}, {"launcher", "singularity"}, {"bind", "/data,/scratch:/s:ro"}, {"containerPythonPath", "/opt/x"}, {"partition", "gpu"}, {"account", "lab"}, {"qos", "normal"}, {"time", "02:00:00"}, {"gpus", 2}, {"cpus", 16}, {"mem", "32G"}, {"port", 7645}, {"engine", true}, {"engineBin", "/opt/e"}, {"perHost", {{"other", {{"partition", "p2"}, {"account", "a2"}, {"qos", ""}, {"time", "00:30:00"}}}}}};
    const nlohmann::json flat = {{"cluster/profile", legacy}, {"cluster/recentFolders", {"/data"}}};
    bool migrated = false;
    cluster::ProfileBook b = cluster::ProfileBook::fromSettings(flat, &migrated);
    CHECK(migrated);
    REQUIRE(b.profiles.size() == 1);
    const cluster::Profile m = b.profiles[0];
    CHECK(m.name == "login.example.org");
    CHECK(b.current == m.name);
    CHECK(m.host == "login.example.org");
    CHECK(m.checkout == "~/src/sirius");
    CHECK(m.container == "/img/w.sif");
    CHECK(m.launcher == "singularity");
    CHECK(m.bind == "/data,/scratch:/s:ro");
    CHECK(m.containerPythonPath == "/opt/x");
    CHECK(m.partition == "gpu");
    CHECK(m.account == "lab");
    CHECK(m.qos == "normal");
    CHECK(m.time == "02:00:00");
    CHECK(m.gpus == 2);
    CHECK(m.cpus == 16);
    CHECK(m.mem == "32G");
    CHECK(m.engine);
    CHECK(m.engineBin == "/opt/e");
    REQUIRE(m.perHost.count("other"));
    CHECK(m.perHost.at("other").partition == "p2");
    // its partition, account and QoS: the dropdowns' first choice; its image the first in its list
    REQUIRE(m.choices.size() == 1);
    CHECK(m.choices[0].name == "gpu");
    CHECK(m.choices[0].isDefault);
    CHECK(m.choices[0].accounts == std::vector<std::string>{"lab"});
    CHECK(m.choices[0].qos == std::vector<std::string>{"normal"});
    CHECK(m.images == std::vector<std::string>{"/img/w.sif"});
    // once there is a profile, the one of before is not read again
    nlohmann::json both = flat;
    for (const auto& [k, v] : b.toSettings()) both[k] = v;
    bool again = true;
    CHECK(cluster::ProfileBook::fromSettings(both, &again).profiles.size() == 1);
    CHECK_FALSE(again);

    // a second cluster, with the choices its dropdowns offer
    cluster::Profile second;
    second.name = "uni";
    second.host = "hpc.uni.example";
    second.container = "/sw/sirius.sif";
    cluster::PartitionChoice c;
    c.name = "short";
    c.accounts = {"a", "b"};
    c.maxTime = "04:00:00";
    c.times = {"01:00:00", "04:00:00"};
    c.maxGpus = 4;
    c.maxMem = "500G";
    second.choices.push_back(c);
    b.put(second);
    CHECK(b.current == "uni");
    const std::map<std::string, nlohmann::json> keys = b.toSettings();
    const std::string text = settings_toml::toToml(nlohmann::json(keys));
    INFO(text);
    CHECK(text.find("login.example.org") != std::string::npos);
    CHECK(text.find("[[cluster.uni.partitions]]") != std::string::npos);
    CHECK(text.find("[cluster.uni.job]") != std::string::npos);
    CHECK(text.find("venv") == std::string::npos);   // the Python environment mode is gone
    const settings_toml::ParseResult back = settings_toml::fromToml(text);
    REQUIRE(back.ok);
    cluster::ProfileBook b2 = cluster::ProfileBook::fromSettings(back.flat);
    REQUIRE(b2.profiles.size() == 2);
    CHECK(b2.current == "uni");
    for (const cluster::Profile& p : b.profiles) {
        REQUIRE(b2.find(p.name));
        CHECK(b2.find(p.name)->toJson() == p.toJson());
    }
    CHECK(cluster::checkClusterSettings(back.flat).empty());

    // names
    CHECK_FALSE(b2.nameProblem("current").empty());
    CHECK_FALSE(b2.nameProblem("a/b").empty());
    CHECK_FALSE(b2.nameProblem("uni").empty());
    CHECK(b2.nameProblem("uni", "uni").empty());
    CHECK(b2.uniqueName("uni") == "uni 2");
    CHECK(b2.uniqueName("current") == "current 2");
    REQUIRE(b2.rename("uni", "university"));
    CHECK(b2.current == "university");
    CHECK(b2.staleKeys(back.flat) == std::vector<std::string>{"cluster/uni"});
    CHECK(b2.remove("university"));
    CHECK(b2.current == "login.example.org");

    // one profile as a small file, and back
    const std::string one = cluster::exportProfile(second);
    const std::vector<cluster::Profile> imported = cluster::importProfiles(one);
    REQUIRE(imported.size() == 1);
    CHECK(imported[0].name == "uni");
    CHECK(imported[0].toJson() == second.toJson());
    CHECK(imported[0].sshProgram.empty());
    CHECK(one.find("ssh =") == std::string::npos);
    bool ignoredSsh = false;
    const std::string withSsh = "[cluster.uni]\nhost = 'hpc.example'\nimage = ''\nssh = 'C:/evil.exe'\n";
    const std::vector<cluster::Profile> dropped = cluster::importProfiles(withSsh, &ignoredSsh);
    CHECK(ignoredSsh);
    REQUIRE(dropped.size() == 1);
    CHECK(dropped[0].sshProgram.empty());
    CHECK(dropped[0].host == "hpc.example");
    CHECK_THROWS_AS(cluster::importProfiles("not = [toml"), std::runtime_error);
    CHECK_THROWS_AS(cluster::importProfiles("[worker]\npython = 'x'\n"), std::runtime_error);
}

TEST_CASE("cluster: the title bar's button says where the session is", "[app][cluster]") {
    using Kind = cluster::ConnectionBadge::Kind;
    const auto now = std::chrono::steady_clock::now();
    cluster::Status st;
    cluster::ConnectionBadge b = cluster::connectionBadge(st, true, now);
    CHECK(b.kind == Kind::Off);
    CHECK(b.label == "Cluster");

    st.state = cluster::State::Connecting;
    st.host = "login.example.org";
    st.steps[0].status = cluster::StepStatus::Done;
    st.steps[1] = cluster::StepState{cluster::StepStatus::Running, "sbatch"};
    b = cluster::connectionBadge(st, true, now);
    CHECK(b.kind == Kind::Connecting);
    CHECK(b.label == "Connecting\xE2\x80\xA6");
    CHECK(std::abs(b.progress - 1.0f / 6.0f) < 1e-6f);
    CHECK(b.tooltip.find("Ask for a job") != std::string::npos);

    st.state = cluster::State::JobReady;
    for (int i = 0; i < cluster::kJobStepCount; ++i) st.steps[static_cast<std::size_t>(i)].status = cluster::StepStatus::Done;
    st.node = "g0003.abc0";
    st.jobId = "4238701";
    st.jobLimitSeconds = 3600;
    st.jobStarted = now - std::chrono::minutes(10);
    b = cluster::connectionBadge(st, true, now);
    CHECK(b.kind == Kind::JobReady);
    CHECK(b.label == "g0003 \xC2\xB7 job 4238701 \xC2\xB7 no worker yet");
    CHECK(b.tooltip.find("50 min left of 1 h 00 min") != std::string::npos);

    st.state = cluster::State::Starting;
    b = cluster::connectionBadge(st, true, now);
    CHECK(b.kind == Kind::Connecting);
    CHECK(b.label == "Starting worker\xE2\x80\xA6");
    CHECK(std::abs(b.progress - 0.5f) < 1e-6f);

    // a worker without SIRIUS's engine: red, nothing runs there
    st.state = cluster::State::Connected;
    st.caps.gpus = {GpuInfo{"NVIDIA A100-SXM4-80GB", 81920}};
    st.caps.cpuThreads = 16;
    b = cluster::connectionBadge(st, true, now);
    CHECK(b.kind == Kind::Failed);
    CHECK(b.label == "g0003 \xC2\xB7 no engine");
    CHECK(b.tooltip.find("no SIRIUS engine") != std::string::npos);
    st.caps.engine = nlohmann::json{{"build", "0.1.0+gabc"}};
    b = cluster::connectionBadge(st, true, now);
    CHECK(b.kind == Kind::Connected);
    CHECK(b.label == "g0003 \xC2\xB7 GPU");
    for (const char* part : {"login.example.org", "4238701", "g0003.abc0", "A100", "50 min left"}) {
        INFO(part);
        CHECK(b.tooltip.find(part) != std::string::npos);
    }
    b = cluster::connectionBadge(st, false, now);
    CHECK(b.label == "g0003 \xC2\xB7 CPU");
    CHECK(b.tooltip.find("16 threads") != std::string::npos);
    st.jobLimitSeconds = -1;
    CHECK(cluster::connectionBadge(st, false, now).tooltip.find("No time limit") != std::string::npos);

    // the worker refused for want of the engine: the job is held, the badge red
    {
        cluster::Status ne = st;
        ne.state = cluster::State::JobReady;
        ne.caps = WorkerCapabilities{};
        ne.noEngine = true;
        ne.reason = "No SIRIUS C++ engine found: the image has no /opt/sirius/bin/sirius-cli, and no Engine builds folder is set.";
        ne.fix = cluster::noEngineFix("/clusterfs/me/sirius-builds");
        const cluster::ConnectionBadge nb = cluster::connectionBadge(ne, true, now);
        CHECK(nb.kind == Kind::Failed);
        CHECK(nb.label == "g0003 \xC2\xB7 no engine");
        CHECK(nb.tooltip.find("e.g. /clusterfs/me/sirius-builds") != std::string::npos);
        // the HPC backend chosen and nothing to run on: "Cluster" and "no worker yet" turn red too
        const std::string why = cluster::wizard::hpcNoEngineReason(cluster::Status{});
        CHECK(why == "HPC: no SIRIUS engine on the cluster \xE2\x80\x94 open Cluster to fix");
        const cluster::ConnectionBadge off = cluster::noEngineBadge(cluster::connectionBadge(cluster::Status{}, true, now), cluster::Status{}, why);
        CHECK(off.kind == Kind::Failed);
        CHECK(off.label == "Cluster \xC2\xB7 no engine");
        CHECK(off.tooltip.rfind(why, 0) == 0);
        ne.noEngine = false;
        const cluster::ConnectionBadge held = cluster::noEngineBadge(cluster::connectionBadge(ne, true, now), ne, cluster::wizard::hpcNoEngineReason(ne));
        CHECK(held.kind == Kind::Failed);
        CHECK(held.label == "g0003 \xC2\xB7 no engine");
        // no reason (the engine answers, or another backend): as it was
        CHECK(cluster::noEngineBadge(cluster::connectionBadge(ne, true, now), ne, "").kind == Kind::JobReady);
    }

    st.state = cluster::State::Disconnected;
    st.dropped = true;
    st.reason = "the SSH connection to login.example.org ended";
    b = cluster::connectionBadge(st, true, now);
    CHECK(b.kind == Kind::Lost);
    CHECK(b.label == "Cluster: lost");
    CHECK(b.tooltip.find("SSH connection") != std::string::npos);
    st.dropped = false;
    st.steps[3] = cluster::StepState{cluster::StepStatus::Failed, "No worker image is set"};
    st.reason = "No worker image is set";
    CHECK(cluster::connectionBadge(st, true, now).label == "Cluster: failed");
    for (cluster::StepState& s : st.steps) s = cluster::StepState{};
    st.reason = "disconnected (job 4238701 left running)";
    CHECK(cluster::connectionBadge(st, true, now).label == "Cluster");

    CHECK(cluster::durationText(59) == "1 min");
    CHECK(cluster::durationText(3600) == "1 h 00 min");
    CHECK(cluster::durationText(90061) == "1 d 1 h");
    CHECK(cluster::durationText(-1).empty());
}

TEST_CASE("cluster: what a change takes, a new job or only a new worker", "[app][cluster]") {
    cluster::Profile a;
    a.host = "h";
    a.container = "/a.sif";
    a.partition = "gpu";
    a.bind = "/x,/y";
    cluster::Profile b = a;
    b.bind = " /x , /y";   // the same folders
    CHECK(cluster::profileChange(a, b).fields.empty());
    b.container = "/b.sif";
    b.bind = "/data";
    cluster::ProfileChange c = cluster::profileChange(a, b);
    CHECK(c.newWorker);
    CHECK_FALSE(c.newJob);
    CHECK(c.fields == std::vector<std::string>{"worker image", "data folders"});
    b.partition = "dgx";
    b.gpus = 4;
    c = cluster::profileChange(a, b);
    CHECK(c.newJob);
    CHECK(std::find(c.fields.begin(), c.fields.end(), "partition") != c.fields.end());
    CHECK(std::find(c.fields.begin(), c.fields.end(), "GPUs") != c.fields.end());
    // what only the dropdowns offer changes nothing that runs
    b = a;
    b.choices.push_back(cluster::PartitionChoice{});
    b.name = "renamed";
    CHECK(cluster::profileChange(a, b).fields.empty());
}

TEST_CASE("cluster: the login fills what a new profile leaves empty; the dropdowns' choices", "[app][cluster]") {
    const cluster::ClusterInfo info = cluster::parseClusterInfo(std::string("@@home /home/tester\n") + kInfoOutput);
    CHECK(info.home == "/home/tester");
    cluster::Profile p;
    const std::vector<std::string> filled = cluster::fillFromCluster(p, info);
    CHECK(p.checkout == "/home/tester/sirius");
    CHECK(p.partition == "cpu");        // sinfo's default
    CHECK(p.gpus == 0);                 // it has none, others have
    CHECK(p.account == "velatkilic");   // the user's association with it
    CHECK(p.qos.empty());               // which names no QoS
    CHECK(filled.size() == 4);
    // what the profile has stays
    cluster::Profile q;
    q.partition = "abc_a100";
    q.account = "abc_lab";
    q.checkout = "/opt/s";
    cluster::fillFromCluster(q, info);
    CHECK(q.partition == "abc_a100");
    CHECK(q.account == "abc_lab");
    CHECK(q.qos == "abc_normal");
    CHECK(q.checkout == "/opt/s");
    CHECK(q.gpus == 1);

    // a partition the cluster reports, kept as a choice
    const cluster::PartitionChoice a100 = cluster::choiceFromCluster(info, "abc_a100");
    CHECK(a100.accounts == std::vector<std::string>{"velatkilic", "abc_lab"});
    CHECK(a100.qos == std::vector<std::string>{"abc_debug", "abc_normal"});
    CHECK(a100.maxTime == "3-00:00:00");
    CHECK(a100.maxGpus == 1);
    CHECK(a100.maxCpus == 32);
    CHECK(a100.maxMem == "488G");
    CHECK_FALSE(a100.isDefault);
    const cluster::PartitionChoice cpu = cluster::choiceFromCluster(info, "cpu");
    CHECK(cpu.isDefault);
    CHECK(cpu.gpus == 0);
    // picking it from the profile's choices
    cluster::Profile r;
    r.account = "someone";
    r.gpus = 4;
    r.time = "5-00:00:00";
    const std::vector<std::string> changed = cluster::applyChoice(r, a100);
    CHECK(r.partition == "abc_a100");
    CHECK(r.account == "velatkilic");
    CHECK(r.qos == "abc_debug");
    CHECK(r.gpus == 1);
    CHECK(r.time == "3-00:00:00");
    CHECK_FALSE(changed.empty());
    // the time limits offered
    using V = std::vector<std::string>;
    CHECK(cluster::timeChoices(&a100, "", "01:00:00") ==
          V{"00:30:00", "01:00:00", "02:00:00", "04:00:00", "08:00:00", "12:00:00", "1-00:00:00", "2-00:00:00", "3-00:00:00"});
    CHECK(cluster::timeChoices(&a100, "01:00:00", "00:45:00") == V{"00:30:00", "00:45:00", "01:00:00"});
    cluster::PartitionChoice own = a100;
    own.times = {"02:00:00"};
    CHECK(cluster::timeChoices(&own, "", "") == V{"02:00:00"});
    CHECK(cluster::timeChoices(nullptr, "", "").size() == 10);
}

TEST_CASE("cluster: the engine build of this application, or one of the same operations, from the builds folder", "[app][cluster]") {
    BuildInfo app = buildInfo();
    app.commit = "1234abcd";
    const nlohmann::json mine = toJson(app);
    nlohmann::json same = mine;
    same["commit"] = "0000other";
    same["build"] = "0.1.0+gother";
    nlohmann::json other = mine;
    other["ops_schema"] = "ffff";
    const std::string out = "builds=yes\n@@build 1234abcd yes " + mine.dump() + "\n@@build 0000other yes " + same.dump() + "\n@@build broken no " +
                            mine.dump() + "\n@@build bad yes {not json\n@@build different yes " + other.dump() + "\n";
    const std::vector<cluster::EngineBuild> found = cluster::parseEngineBuilds(out);
    REQUIRE(found.size() == 5);
    CHECK(found[0].dir == "1234abcd");
    CHECK(found[0].runnable);
    CHECK(found[0].readable);
    CHECK_FALSE(found[2].runnable);
    CHECK_FALSE(found[3].readable);
    std::string note;
    CHECK(cluster::pickEngineBuild(found, app, &note) == 0);
    CHECK(note == "this build");
    // without its own: the newest of the same operations
    const std::vector<cluster::EngineBuild> rest(found.begin() + 1, found.end());
    CHECK(cluster::pickEngineBuild(rest, app, &note) == 0);
    CHECK(note == "the same operations as this build");
    // never one that cannot run, cannot be read, or has other operations
    CHECK(cluster::pickEngineBuild({found[2], found[3], found[4]}, app, &note) == -1);
    // a matching build that is not whole is named first: its fix is a reinstall, not a new build
    CHECK(note.find("matches this application but has no bin/sirius-cli") != std::string::npos);
    CHECK(note.find("3 builds checked") != std::string::npos);
    CHECK(cluster::pickEngineBuild({found[3], found[4]}, app, &note) == -1);
    CHECK(note.find("None of the 2 engine builds can be used") != std::string::npos);
    CHECK(cluster::pickEngineBuild({}, app, &note) == -1);
    CHECK(note.find("holds no build") != std::string::npos);
    CHECK(cluster::engineBuildsScript("~/engines").find("BUILD.json") != std::string::npos);
}

// --- two steps: the job, then the worker in it ---------------------------------------------

TEST_CASE("cluster: the job first, then the worker in it; a new image restarts only the worker; another partition takes a new job",
          "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the worker");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "w.sif") << "image\n";
    std::ofstream(fc.home / "w2.sif") << "image\n";
    setEnv("FAKE_CONTAINER_SITE", containerSite(fc).string());
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    const cluster::Profile p = containerProfile(fc, "~/w.sif");

    // step 1: the job, nothing of SIRIUS run in it yet
    session.connectJob(p);
    REQUIRE(waitFor([&] { return settled(session); }, std::chrono::seconds(120)));
    cluster::Status st = session.status();
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::JobReady);
    CHECK(st.jobId == "4711");
    CHECK(st.node == "fakenode");
    CHECK(st.jobLimitSeconds == 3600);   // squeue's %l
    for (int i = 0; i < cluster::kStepCount; ++i)
        CHECK(st.steps[static_cast<std::size_t>(i)].status == (i < cluster::kJobStepCount ? cluster::StepStatus::Done : cluster::StepStatus::Pending));
    CHECK(noWorkerStarted(fc));
    CHECK(cluster::connectionBadge(st, false, std::chrono::steady_clock::now()).label == "fakenode \xC2\xB7 job 4711 \xC2\xB7 no worker yet");
    CHECK(session.hasJobRunning());
    CHECK_FALSE(session.connected());
    // Connect while it holds: no second job
    session.connectJob(p);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CHECK(session.status().state == cluster::State::JobReady);
    CHECK_FALSE(fs::exists(fc.slurm / "4712.args"));

    // step 2: the worker, as a step of that job
    st = workerUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    CHECK(st.jobId == "4711");
    const std::string steps = readAll(fc.slurm / "srun.args");
    CHECK(countOf(steps, "--job-name=sirius-worker") == 1);
    // the checks ran in the job before it, through srun, with the job's id
    CHECK(countOf(steps, "--job-name=sirius-check") == 1);
    CHECK(steps.find("--jobid=4711 --overlap --nodes=1 --ntasks=1 --job-name=sirius-check bash ") != std::string::npos);
    CHECK(steps.find(cluster::workerLaunchScriptName() + " --check") != std::string::npos);
    CHECK(steps.find("--job-name=sirius-check") < steps.find("--job-name=sirius-worker"));
    CHECK(steps.find("--jobid=4711 --overlap") != std::string::npos);

    // a new image: the worker again, in the same job
    cluster::Profile q = p;
    q.container = "~/w2.sif";
    const cluster::ProfileChange change = cluster::profileChange(session.profile(), q);
    CHECK(change.newWorker);
    CHECK_FALSE(change.newJob);
    st = workerUntilSettled(session, q);
    INFO(st.reason);
    REQUIRE(st.state == cluster::State::Connected);
    CHECK(st.jobId == "4711");
    CHECK_FALSE(fs::exists(fc.slurm / "4712.args"));
    CHECK(countOf(readAll(fc.slurm / "srun.args"), "--job-name=sirius-worker") == 2);
    CHECK(countOf(readAll(fc.slurm / "srun.args"), "--job-name=sirius-check") == 2);   // checked again before it
    CHECK(readAll(fc.slurm / "cancelled_steps").find("4711.") != std::string::npos);   // the first worker's step ended
    CHECK(readAll(fc.slurm / "4711.env").find("w2.sif") != std::string::npos);
    CHECK(fs::exists(fc.home / ".sirius" / "run" / "sirius-worker-4711-2.log"));

    // the worker stopped: the job stays
    session.stopWorker();
    st = session.status();
    CHECK(st.state == cluster::State::JobReady);
    CHECK(st.jobId == "4711");
    CHECK(session.hasJobRunning());

    // another partition: a new job, once the old one is let go
    cluster::Profile r = q;
    r.partition = "dgx";
    CHECK(cluster::profileChange(session.profile(), r).newJob);
    // Change job: the job cancelled, the login kept (no second ssh)
    const auto logins = countOf(fc.sshLog(), "argv ");
    session.cancelJob();
    CHECK(readAll(fc.slurm / "cancelled").find("4711") != std::string::npos);
    st = session.status();
    CHECK(st.state == cluster::State::Idle);
    CHECK(st.sshUp);
    CHECK(st.jobId.empty());
    CHECK(st.node.empty());
    CHECK(st.steps[0].status == cluster::StepStatus::Done);
    CHECK(st.steps[1].status == cluster::StepStatus::Pending);
    CHECK(cluster::wizard::loggedIn(st, p.host));
    CHECK(cluster::wizard::startJobGate(st, p.host).enabled);
    CHECK_FALSE(session.hasJobRunning());
    session.connectJob(r);
    REQUIRE(waitFor([&] { return settled(session); }, std::chrono::seconds(120)));
    st = session.status();
    CHECK(st.state == cluster::State::JobReady);
    CHECK(st.jobId == "4712");
    CHECK(readAll(fc.slurm / "4712.args").find("--partition=dgx") != std::string::npos);
    CHECK(countOf(fc.sshLog(), "argv ") == logins);   // the same login
    session.disconnect(true);
    setEnv("FAKE_CONTAINER_SITE", "");
}

TEST_CASE("cluster: an image is built in the held job where the cluster allows it, and refused in words where not", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    copyCheckout(fc.home / "sirius");
    fs::create_directories(fc.home / "sirius" / "containers");
    std::ofstream(fc.home / "sirius" / "containers" / "sirius-worker.def") << "Bootstrap: docker\nFrom: python:3.12\n";
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    session.connectJob(containerProfile(fc, ""));
    REQUIRE(waitFor([&] { return settled(session); }, std::chrono::seconds(120)));
    REQUIRE(session.status().state == cluster::State::JobReady);
    REQUIRE(session.clusterInfo());
    const std::string home = session.clusterInfo()->home;
    REQUIRE_FALSE(home.empty());
    const cluster::Profile p = session.profile();
    auto buildSettled = [&] {
        const cluster::BuildStatus::Phase ph = session.status().build.phase;
        return ph != cluster::BuildStatus::Phase::Probing && ph != cluster::BuildStatus::Phase::Building;
    };

    // from the checkout's definition file, as a step of the job
    session.buildImage(p, "", home + "/new.sif");
    REQUIRE(waitFor([&] { return session.status().build.phase != cluster::BuildStatus::Phase::None && buildSettled(); }, std::chrono::seconds(120)));
    cluster::BuildStatus b = session.status().build;
    INFO(b.error << "\n"
                 << b.log << "\n"
                 << b.why);
    CHECK(b.phase == cluster::BuildStatus::Phase::Done);
    REQUIRE(b.supported);
    CHECK(*b.supported);
    CHECK(b.image == home + "/new.sif");
    CHECK(readAll(fc.home / "new.sif").find("sirius-worker.def") != std::string::npos);
    CHECK(b.log.find("Build complete") != std::string::npos);
    CHECK(readAll(fc.slurm / "srun.args").find("--jobid=4711") != std::string::npos);
    // an existing file is never written over
    session.buildImage(p, "", home + "/new.sif");
    REQUIRE(waitFor([&] { return session.status().build.phase == cluster::BuildStatus::Phase::Failed; }, std::chrono::seconds(120)));
    CHECK(session.status().build.error.find("already") != std::string::npos);

    // a cluster without fakeroot: said, and nothing built
    std::ofstream(fc.slurm / "no_fakeroot") << "1\n";
    session.buildImage(p, "", home + "/other.sif");
    REQUIRE(waitFor([&] { return session.status().build.supported.has_value() && !*session.status().build.supported && buildSettled(); },
                    std::chrono::seconds(120)));
    b = session.status().build;
    CHECK(b.phase == cluster::BuildStatus::Phase::Failed);
    CHECK(b.error.find("does not let you build images") != std::string::npos);
    CHECK(b.why.find("fakeroot") != std::string::npos);
    CHECK_FALSE(fs::exists(fc.home / "other.sif"));
    session.disconnect(true);
}

TEST_CASE("cluster wizard: each page's Next waits for what the page is for; the dialog opens where the session is", "[app][cluster]") {
    namespace wz = cluster::wizard;
    using wz::Page;
    cluster::Status st;
    const std::string host = "fiona";

    // nothing yet: page 1, Connect needs a host, Next waits for the login
    CHECK(wz::openingPage(st, host) == Page::Connect);
    CHECK_FALSE(wz::connectGate(st, "").enabled);
    CHECK(wz::connectGate(st, host).enabled);
    CHECK_FALSE(wz::nextGate(Page::Connect, st, host).enabled);
    CHECK(wz::nextGate(Page::Connect, st, host).why.find("Connect first") != std::string::npos);
    CHECK_FALSE(wz::startJobGate(st, host).enabled);
    CHECK_FALSE(wz::pageDone(Page::Connect, st, host));

    // logging in
    st.state = cluster::State::Connecting;
    st.steps[0].status = cluster::StepStatus::Running;
    CHECK(wz::openingPage(st, host) == Page::Connect);
    CHECK_FALSE(wz::nextGate(Page::Connect, st, host).enabled);
    CHECK_FALSE(wz::connectGate(st, host).enabled);

    // logged in (Idle again, the SSH session up for this host): Next, Start job
    st.state = cluster::State::Idle;
    st.steps[0].status = cluster::StepStatus::Done;
    st.sshUp = true;
    st.host = host;
    CHECK(wz::nextGate(Page::Connect, st, host).enabled);
    CHECK(wz::pageDone(Page::Connect, st, host));
    CHECK_FALSE(wz::nextGate(Page::Connect, st, "other").enabled);   // logged in to another host
    CHECK(wz::startJobGate(st, host).enabled);
    CHECK_FALSE(wz::nextGate(Page::Job, st, host).enabled);
    CHECK_FALSE(wz::startWorkerGate(st, "/x.sif").enabled);

    // the job in the queue: page 2, Next waits
    st.state = cluster::State::Connecting;
    st.steps[1].status = cluster::StepStatus::Done;
    st.steps[2] = cluster::StepState{cluster::StepStatus::Running, "job 4711 \xC2\xB7 PENDING (Resources) \xC2\xB7 0:42"};
    st.jobId = "4711";
    CHECK(wz::openingPage(st, host) == Page::Job);
    CHECK_FALSE(wz::nextGate(Page::Job, st, host).enabled);
    CHECK_FALSE(wz::startJobGate(st, host).enabled);

    // the job runs: Next; the worker needs an image
    st.state = cluster::State::JobReady;
    st.steps[2] = cluster::StepState{cluster::StepStatus::Done, "job 4711 runs on g0003"};
    st.node = "g0003";
    CHECK(wz::openingPage(st, host) == Page::Job);   // a reattached job shows on its page
    CHECK(wz::nextGate(Page::Job, st, host).enabled);
    CHECK(wz::pageDone(Page::Job, st, host));
    CHECK_FALSE(wz::startJobGate(st, host).enabled);     // Change job… instead
    CHECK_FALSE(wz::connectGate(st, "other").enabled);   // tied to its host until disconnected
    CHECK_FALSE(wz::startWorkerGate(st, "").enabled);
    CHECK(wz::startWorkerGate(st, "").why.find("image") != std::string::npos);
    CHECK(wz::startWorkerGate(st, "/x.sif").enabled);
    CHECK_FALSE(wz::nextGate(Page::Worker, st, host).enabled);

    // the worker starting, then failed: page 3, Finish waits
    st.state = cluster::State::Starting;
    CHECK(wz::openingPage(st, host) == Page::Worker);
    CHECK_FALSE(wz::nextGate(Page::Worker, st, host).enabled);
    CHECK_FALSE(wz::startWorkerGate(st, "/x.sif").enabled);
    st.state = cluster::State::JobReady;
    st.steps[3] = cluster::StepState{cluster::StepStatus::Failed, "No worker image is set"};
    CHECK(wz::openingPage(st, host) == Page::Worker);
    CHECK(wz::nextGate(Page::Worker, st, host).why.find("did not start") != std::string::npos);

    // a worker without SIRIUS's engine is not ready: no Finish, the dialog stays on the Worker page
    st.state = cluster::State::Connected;
    for (int i = 3; i < cluster::kStepCount; ++i) st.steps[static_cast<std::size_t>(i)].status = cluster::StepStatus::Done;
    CHECK_FALSE(wz::nextGate(Page::Worker, st, host).enabled);
    CHECK(wz::nextGate(Page::Worker, st, host).why.find("No SIRIUS engine") != std::string::npos);
    CHECK_FALSE(wz::pageDone(Page::Worker, st, host));
    CHECK(wz::openingPage(st, host) == Page::Worker);
    CHECK_FALSE(wz::engineReady(st));

    // SIRIUS's engine answers: Finish; the dialog opens on the summary
    st.caps.engine = nlohmann::json{{"build", "0.1.0+gabc"}};
    CHECK(wz::engineReady(st));
    CHECK(wz::hpcNoEngineReason(st).empty());
    CHECK(wz::nextGate(Page::Worker, st, host).enabled);
    CHECK(wz::pageDone(Page::Worker, st, host));
    CHECK(wz::openingPage(st, host) == Page::Summary);

    // a login that failed: page 1 again
    cluster::Status failed;
    failed.state = cluster::State::Disconnected;
    failed.steps[0] = cluster::StepState{cluster::StepStatus::Failed, "SSH login to fiona failed"};
    CHECK(wz::openingPage(failed, host) == Page::Connect);
    // sbatch refused, still logged in: the job page
    cluster::Status refused;
    refused.state = cluster::State::Disconnected;
    refused.sshUp = true;
    refused.host = host;
    refused.steps[0].status = cluster::StepStatus::Done;
    refused.steps[1] = cluster::StepState{cluster::StepStatus::Failed, "sbatch refused the job."};
    refused.reason = "sbatch refused the job.";
    CHECK(wz::openingPage(refused, host) == Page::Job);
    CHECK(wz::startJobGate(refused, host).enabled);
}

TEST_CASE("cluster wizard: the login's outcome in plain words, ssh's own output kept for Details", "[app][cluster]") {
    namespace wz = cluster::wizard;
    using K = wz::LoginOutcome::Kind;
    cluster::Status st;
    CHECK(wz::loginOutcome(st, "fiona", "").kind == K::None);
    st.sshUp = true;
    st.host = "fiona";
    wz::LoginOutcome o = wz::loginOutcome(st, "fiona", "velat");
    CHECK(o.kind == K::Ok);
    CHECK(o.text == "Connected to fiona as velat");

    st = cluster::Status{};
    st.state = cluster::State::Disconnected;
    st.steps[0] = cluster::StepState{cluster::StepStatus::Failed, "SSH login to fiona failed: the SSH connection ended."};
    st.reason = st.steps[0].detail;
    st.remoteOutput = "velat@fiona: Permission denied (keyboard-interactive).";
    o = wz::loginOutcome(st, "fiona", "");
    CHECK(o.kind == K::Failed);
    CHECK(o.text.rfind("Wrong password", 0) == 0);
    CHECK(o.details == st.remoteOutput);

    const auto words = [](const std::string& out) {
        return wz::loginFailureWords("fiona", "SSH login to fiona failed: the SSH connection ended.", out);
    };
    CHECK(words("ssh: Could not resolve hostname fiona: No such host is known.").rfind("Host not found", 0) == 0);
    CHECK(words("ssh: connect to host fiona port 22: Connection timed out").rfind("No answer from fiona (timed out)", 0) == 0);
    CHECK(words("ssh: connect to host fiona port 22: Connection refused").rfind("fiona refused the connection", 0) == 0);
    CHECK(words("ssh: connect to host fiona port 22: Network is unreachable").find("cannot reach fiona") != std::string::npos);
    CHECK(words("Host key verification failed.").find("host key") != std::string::npos);
    CHECK(words("Received disconnect from 1.2.3.4: Too many authentication failures").rfind("Too many keys", 0) == 0);
    CHECK(words("").find("closed the connection") != std::string::npos);   // the session's own words
    CHECK(wz::loginFailureWords("fiona", "Login cancelled: nothing was sent for the prompt you closed.", "") ==
          "Login cancelled: nothing was sent for the prompt you closed.");
    CHECK(wz::loginFailureWords("fiona", "something else", "") == "something else");
}

TEST_CASE("cluster wizard: the job's line says queued and why, then where it runs and the time left", "[app][cluster]") {
    namespace wz = cluster::wizard;
    using K = wz::JobLine::Kind;
    const auto now = std::chrono::steady_clock::now();
    cluster::Status st;
    CHECK(wz::jobLine(st, now).kind == K::None);
    st.state = cluster::State::Connecting;
    st.steps[0].status = cluster::StepStatus::Done;
    st.steps[1] = cluster::StepState{cluster::StepStatus::Running, "sbatch"};
    CHECK(wz::jobLine(st, now).text == "Submitting the job\xE2\x80\xA6");
    st.steps[1].status = cluster::StepStatus::Done;
    st.jobId = "4711";
    st.steps[2] = cluster::StepState{cluster::StepStatus::Running, "job 4711 \xC2\xB7 PENDING (Resources) \xC2\xB7 0:42"};
    wz::JobLine l = wz::jobLine(st, now);
    CHECK(l.kind == K::Busy);
    CHECK(l.text == "Job 4711 queued (waiting for Resources) \xC2\xB7 waited 0:42");
    st.state = cluster::State::JobReady;
    st.node = "g0003.abc0";
    st.jobLimitSeconds = 3600;
    st.jobStarted = now - std::chrono::minutes(10);
    l = wz::jobLine(st, now);
    CHECK(l.kind == K::Running);
    CHECK(l.text == "Running on g0003.abc0 \xC2\xB7 job 4711 \xC2\xB7 50 min left");
    st.jobLimitSeconds = -1;
    CHECK(wz::timeLeftText(st, now) == "no time limit");

    cluster::Status refused;
    refused.state = cluster::State::Disconnected;
    refused.steps[1] = cluster::StepState{cluster::StepStatus::Failed, "sbatch refused the job."};
    refused.reason = "sbatch refused the job.";
    l = wz::jobLine(refused, now);
    CHECK(l.kind == K::Failed);
    CHECK(l.text == "sbatch refused the job.");
}

TEST_CASE("cluster wizard: the worker's health report from its hello", "[app][cluster]") {
    namespace wz = cluster::wizard;
    using M = wz::Mark;
    using V = wz::HealthReport::Verdict;
    const auto now = std::chrono::steady_clock::now();
    BuildInfo app;
    app.version = "0.1.0";
    app.build = "0.1.0+gabc";
    app.commit = "abcdef0123456789";
    app.opsSchema = "ops1";
    cluster::Profile p;
    p.gpus = 1;
    p.cpus = 8;
    p.mem = "64G";
    p.bind = "/data,/scratch/me";
    p.engine = true;

    const auto find = [](const wz::HealthReport& r, const std::string& label) -> const wz::HealthRow* {
        for (const wz::HealthRow& row : r.rows)
            if (row.label == label) return &row;
        return nullptr;
    };

    // no worker yet: the steps it goes through
    cluster::Status st;
    st.state = cluster::State::JobReady;
    st.jobId = "4711";
    st.node = "g0003.abc0";
    wz::HealthReport r = wz::healthReport(st, p, app, now);
    CHECK(r.verdict == V::None);
    REQUIRE(find(r, "Checks"));
    CHECK(find(r, "Checks")->value == "not yet");

    // it failed: the reason, the fix, the cluster's words
    st.steps[3] = cluster::StepState{cluster::StepStatus::Failed, "The worker image /x.sif is not there."};
    st.reason = "The worker image /x.sif is not there.";
    st.fix = "Pick the image under Worker image.";
    st.remoteOutput = "ls: /x.sif: No such file";
    r = wz::healthReport(st, p, app, now);
    CHECK(r.verdict == V::Failed);
    CHECK(r.headline == "Not ready: The worker image /x.sif is not there.");
    CHECK(r.fix == st.fix);
    CHECK(r.details == st.remoteOutput);
    CHECK(find(r, "Checks")->mark == M::Fail);

    // the engine's hello
    st = cluster::Status{};
    st.state = cluster::State::Connected;
    st.jobId = "4711";
    st.node = "g0003.abc0";
    st.jobLimitSeconds = 3600;
    st.jobStarted = now - std::chrono::minutes(10);
    st.engineBuild = "/home/me/engines/abcdef0";
    st.caps.gpus = {GpuInfo{"NVIDIA A100-SXM4-80GB", 81920}};
    st.caps.cuda = true;
    st.caps.cudaUsable = true;
    st.caps.device = "cuda:0 \xC2\xB7 NVIDIA A100-SXM4-80GB \xC2\xB7 80 GB";
    st.caps.cpuThreads = 64;
    st.caps.tiffReader = "0.1.0";
    nlohmann::json engine = toJson(app);
    engine["cuda"] = {{"devices", nlohmann::json::array()}, {"nvtiff", true}};
    engine["python"] = {{"state", "ready"}, {"caps", {{"torch", "2.5.1+cu124"}}}};
    st.caps.engine = engine;
    r = wz::healthReport(st, p, app, now);
    INFO(r.headline);
    CHECK(r.verdict == V::Ready);
    CHECK(r.headline == "Ready");
    CHECK(find(r, "Node")->value == "g0003.abc0 \xC2\xB7 job 4711");
    CHECK(find(r, "GPU")->value == "1\xC3\x97 A100 80 GB");
    CHECK(find(r, "CUDA")->mark == M::Ok);
    CHECK(find(r, "torch")->value == "2.5.1+cu124");
    CHECK(find(r, "sirius package")->value == "0.1.0");
    CHECK(find(r, "nvTIFF")->mark == M::Ok);
    const wz::HealthRow* e = find(r, "C++ engine");
    REQUIRE(e);
    CHECK(e->mark == M::Ok);
    CHECK(e->value.find("0.1.0+gabc (abcdef0123)") != std::string::npos);
    CHECK(e->value.find("this application's build") != std::string::npos);
    CHECK(e->value.find("from /home/me/engines/abcdef0") != std::string::npos);
    CHECK(find(r, "CPU threads")->value.find("64 on the node") != std::string::npos);
    CHECK(find(r, "Memory")->value == "64G (the job's)");
    CHECK(find(r, "Data folders")->value == "/data, /scratch/me (and your home folder)");
    CHECK(find(r, "Job time left")->value == "50 min left");

    // another commit of the same operations; an engine of other operations
    engine["commit"] = "1234567";
    st.caps.engine = engine;
    CHECK(find(wz::healthReport(st, p, app, now), "C++ engine")->value.find("compatible (same operations)") != std::string::npos);
    engine["ops_schema"] = "ops2";
    st.caps.engine = engine;
    r = wz::healthReport(st, p, app, now);
    CHECK(find(r, "C++ engine")->mark == M::Fail);
    CHECK(r.headline == "Ready, with 1 warning");

    // the Python worker alone, without CUDA, the sirius package or torch: not
    // ready at all (nothing runs on HPC without the engine), the engine row
    // failed with what to do
    st.caps = WorkerCapabilities{};
    st.caps.gpus = {GpuInfo{"NVIDIA A100-SXM4-80GB", 81920}};
    st.caps.cudaReason = "no CUDA library in the worker's environment";
    st.caps.device = "cpu \xC2\xB7 8 threads";
    r = wz::healthReport(st, p, app, now);
    CHECK(r.verdict == V::Failed);
    CHECK(r.headline.rfind("Not ready: no SIRIUS C++ engine", 0) == 0);
    CHECK(r.fix.find("Engine builds folder (Job \xE2\x96\xB8 More options)") != std::string::npos);
    CHECK(find(r, "CUDA")->value == "not usable: no CUDA library in the worker's environment");
    CHECK(find(r, "CUDA")->mark == M::Warn);
    CHECK(find(r, "torch")->mark == M::Warn);
    CHECK(find(r, "sirius package")->mark == M::Warn);
    CHECK(find(r, "C++ engine")->mark == M::Fail);
    CHECK(find(r, "C++ engine")->value.find("Set Engine builds folder") != std::string::npos);
    CHECK(find(r, "nvTIFF")->mark == M::Info);
    {
        cluster::Profile off = p;
        off.engine = false;
        const wz::HealthReport ro = wz::healthReport(st, off, app, now);
        CHECK(ro.verdict == V::Failed);
        CHECK(find(ro, "C++ engine")->value.find("Run SIRIUS's C++ engine on the node") != std::string::npos);
    }
    // the engine not found by the checks: the report's engine row says so, with the fix
    {
        cluster::Status ne;
        ne.state = cluster::State::JobReady;
        ne.jobId = "4711";
        ne.node = "g0003.abc0";
        ne.noEngine = true;
        ne.steps[3] = cluster::StepState{cluster::StepStatus::Failed, "No SIRIUS C++ engine found"};
        ne.reason = "No SIRIUS C++ engine found: the image has no /opt/sirius/bin/sirius-cli, and no Engine builds folder is set.";
        ne.fix = cluster::noEngineFix("/clusterfs/nvme2/Users/velatkilic/sirius-builds");
        const wz::HealthReport rn = wz::healthReport(ne, p, app, now);
        CHECK(rn.verdict == V::Failed);
        CHECK(rn.headline.find("No SIRIUS C++ engine found") != std::string::npos);
        REQUIRE(find(rn, "C++ engine"));
        CHECK(find(rn, "C++ engine")->mark == M::Fail);
        CHECK(find(rn, "C++ engine")->value.find("e.g. /clusterfs/nvme2/Users/velatkilic/sirius-builds") != std::string::npos);
        CHECK(wz::hpcNoEngineReason(ne) == "HPC: no SIRIUS engine in job 4711 \xE2\x80\x94 open Cluster to fix");
    }
    st.caps.torch = "2.4.0";
    CHECK(find(wz::healthReport(st, p, app, now), "torch")->value == "2.4.0");
    // no GPU asked for: none is not a warning
    p.gpus = 0;
    st.caps.gpus.clear();
    r = wz::healthReport(st, p, app, now);
    CHECK(find(r, "GPU")->mark == M::Info);
    CHECK(find(r, "CUDA")->mark == M::Info);
    // little time left
    st.jobStarted = now - std::chrono::minutes(55);
    CHECK(find(wz::healthReport(st, p, app, now), "Job time left")->mark == M::Warn);
}

// --- the launch script is the application's; the engine is never silently missing ---------------

namespace {

    // The application's own launch script, as a file (LF only, executable).
    fs::path writeAppScript(const fs::path& path) {
        writeScript(path, cluster::workerLaunchScript());
        return path;
    }

    // A stand-in "engine" that is the Python worker alone: what an image
    // without SIRIUS's engine runs when the engine is asked for.
    fs::path pythonOnlyEngine(const FakeCluster& fc) {
        const fs::path p = fc.home / "bin" / "not-an-engine";
        fs::create_directories(p.parent_path());
        // it even answers `version` as this build: only its hello gives it away
        const nlohmann::json version = {{"command", "version"}, {"ok", true}, {"result", {{"name", "sirius-cli"}, {"build", toJson(buildInfo())}}}};
        writeScript(p, "#!/bin/bash\nif [ \"$1\" = version ]; then printf '%s\\n' '" + version.dump() +
                           "'; exit 0; fi\nargs=()\nfor a in \"$@\"; do case \"$a\" in serve|--no-python-worker) ;; *) args+=(\"$a\") ;; esac; done\n"
                           "exec \"$FAKE_PYTHON\" -m sirius_worker \"${args[@]}\"\n");
        return p;
    }

    // <folder>/<dir>: a per-commit engine build as the cluster agent makes it
    // (bin/sirius-cli, lib/, python/sirius_worker, BUILD.json), its
    // executable this build's sirius-cli behind a script.
    fs::path engineBuild(const fs::path& folder, const std::string& dir, const nlohmann::json& buildJson) {
        const fs::path b = folder / dir;
        fs::create_directories(b / "bin");
        fs::create_directories(b / "lib");
        // `version` answers with the build of BUILD.json, as a real build's does
        const nlohmann::json version = {{"command", "version"}, {"ok", true}, {"result", {{"name", "sirius-cli"}, {"build", buildJson}}}};
        writeScript(b / "bin" / "sirius-cli", "#!/bin/bash\nif [ \"$1\" = version ]; then printf '%s\\n' '" + version.dump() + "'; exit 0; fi\nexec \"" +
                                                  std::string(SIRIUS_TEST_CLI) + "\" \"$@\"\n");
        fs::create_directories(b / "python");
        fs::copy(fs::path(SIRIUS_TEST_SOURCE_DIR) / "app" / "python" / "sirius_worker", b / "python" / "sirius_worker", fs::copy_options::recursive);
        std::ofstream(b / "BUILD.json") << buildJson.dump();
        return b;
    }

} // namespace

TEST_CASE("cluster: the worker runs the application's own launch script, never the checkout's", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the worker");
    // what is compiled in is the file of this tree, line endings aside
    std::string source = readAll(fs::path(SIRIUS_TEST_SOURCE_DIR) / "app" / "python" / "slurm" / "sirius_worker.sbatch");
    source.erase(std::remove(source.begin(), source.end(), '\r'), source.end());
    CHECK(cluster::workerLaunchScript() == source);
    CHECK(cluster::workerLaunchScript().find("SIRIUS_ENGINE") != std::string::npos);
    CHECK(cluster::workerLaunchScriptName().rfind("sirius_worker-", 0) == 0);
    CHECK(cluster::workerLaunchScriptName().find(buildInfo().version) != std::string::npos);
    CHECK(cluster::workerLaunchScriptName().find('/') == std::string::npos);

    // a checkout older than the application: its script knows no engine
    copyCheckout(fc.home / "sirius");
    const fs::path old = fc.home / "sirius" / "app" / "python" / "slurm" / "sirius_worker.sbatch";
    writeScript(old, "#!/bin/bash\necho OUTDATED-CHECKOUT-SCRIPT\nexec python -m sirius_worker --host 0.0.0.0 --port 0\n");
    std::ofstream(fc.home / "w.sif") << "image\n";
    setEnv("FAKE_CONTAINER_SITE", containerSite(fc).string());
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    const cluster::Status st = connectUntilSettled(session, containerProfile(fc, "~/w.sif"));
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    // the step ran the script the application wrote to ~/.sirius/run, never the checkout's
    const std::string steps = readAll(fc.slurm / "srun.args");
    CHECK(steps.find(cluster::workerLaunchScriptName()) != std::string::npos);
    CHECK(steps.find("app/python/slurm/sirius_worker.sbatch") == std::string::npos);
    std::string ran = readAll(fc.slurm / "4711.script");
    while (!ran.empty() && (ran.back() == '\n' || ran.back() == '\r')) ran.pop_back();
    CHECK(ran.find("/.sirius/run/" + cluster::workerLaunchScriptName()) != std::string::npos);
    const std::string text = readAll(fc.slurm / "4711.script.txt");
    CHECK(text == cluster::workerLaunchScript());
    CHECK(text.find("OUTDATED-CHECKOUT-SCRIPT") == std::string::npos);
    const fs::path written = fc.home / ".sirius" / "run" / cluster::workerLaunchScriptName();
    REQUIRE(fs::exists(written));
    CHECK(readAll(written) == cluster::workerLaunchScript());
#ifndef _WIN32
    CHECK((fs::status(written).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);
    CHECK((fs::status(written).permissions() & fs::perms::owner_exec) != fs::perms::none);
#endif
    // the worker's code was named outright: the checkout's app/python
    const std::string env = readAll(fc.slurm / "4711.env");
    CHECK(env.find("SIRIUS_WORKER_DIR=") != std::string::npos);
    CHECK(env.find("sirius/app/python") != std::string::npos);
    CHECK(st.steps[static_cast<int>(cluster::Step::Checks)].detail.rfind("checkout", 0) == 0);
    session.disconnect(true);
    setEnv("FAKE_CONTAINER_SITE", "");
}

TEST_CASE("cluster: the engine asked for and not found fails the checks with the fix, and no worker starts", "[app][cluster][engine]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the checks");
    copyCheckout(fc.home / "sirius");
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    // per-commit builds exist on the cluster, the profile does not name them: the fix does
    nlohmann::json other = toJson(buildInfo());
    other["commit"] = "8368aab4582f71273b16eaa63ce09d38ba867d8e";
    engineBuild(fc.home / "sirius-builds", "8368aab4582f71273b16eaa63ce09d38ba867d8e", other);
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    cluster::Profile p = engineProfile(fc);
    p.engineBin.clear();   // the image's own engine, which this image does not have
    cluster::Status st = connectUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    CHECK(st.state == cluster::State::JobReady);
    CHECK(st.steps[static_cast<int>(cluster::Step::Checks)].status == cluster::StepStatus::Failed);
    CHECK(st.noEngine);
    CHECK(st.reason.rfind("No SIRIUS C++ engine found", 0) == 0);
    CHECK(st.fix.find("Set Engine builds folder (Job \xE2\x96\xB8 More options) to the folder holding per-commit builds, e.g. ") != std::string::npos);
    CHECK(st.fix.find("sirius-builds") != std::string::npos);   // the one the checks found
    CHECK(noWorkerStarted(fc));                                  // never the Python worker instead
    const cluster::wizard::HealthReport r = cluster::wizard::healthReport(st, p, buildInfo(), std::chrono::steady_clock::now());
    CHECK(r.verdict == cluster::wizard::HealthReport::Verdict::Failed);
    bool engineRow = false;
    for (const cluster::wizard::HealthRow& row : r.rows)
        if (row.label == "C++ engine") engineRow = row.mark == cluster::wizard::Mark::Fail && row.value.find("Engine builds folder") != std::string::npos;
    CHECK(engineRow);

    // an executable named outright that is not there: the same
    p.engineBin = "~/nowhere/sirius-cli";
    st = workerUntilSettled(session, p);
    CHECK(st.steps[static_cast<int>(cluster::Step::Checks)].status == cluster::StepStatus::Failed);
    CHECK(st.noEngine);
    CHECK(st.reason.find("~/nowhere/sirius-cli is neither an executable") != std::string::npos);
    CHECK(noWorkerStarted(fc));

    // the builds folder named: the build of the same operations is taken
    // (its BUILD.json with schema_hash only, not ops_schema), its python/ is
    // the worker's code, and its engine answers
    nlohmann::json eightK = toJson(buildInfo());
    eightK.erase("ops_schema");
    eightK["schema_hash"] = buildInfo().opsSchema;
    eightK["commit"] = "8368aab4582f71273b16eaa63ce09d38ba867d8e";
    eightK["build"] = "0.1.0+g8368aab";
    fs::remove_all(fc.home / "sirius-builds");
    engineBuild(fc.home / "sirius-builds", "8368aab4582f71273b16eaa63ce09d38ba867d8e", eightK);
    p.engineBin.clear();
    p.engineBuilds = "~/sirius-builds";
    st = workerUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    CHECK_FALSE(st.noEngine);
    CHECK(st.engineBuild.find("8368aab4582f71273b16eaa63ce09d38ba867d8e") != std::string::npos);
    if (buildInfo().commit != "8368aab4582f71273b16eaa63ce09d38ba867d8e") CHECK(st.engineBuildNote == "the same operations as this build");
    CHECK(st.steps[static_cast<int>(cluster::Step::Checks)].detail.rfind("worker's code from the engine build", 0) == 0);
    const std::string env = readAll(fc.slurm / "4711.env");
    CHECK(env.find("SIRIUS_WORKER_DIR=") != std::string::npos);
    CHECK(env.find("8368aab4582f71273b16eaa63ce09d38ba867d8e/python") != std::string::npos);
    CHECK(env.find("SIRIUS_ENGINE_DIR=") != std::string::npos);
    CHECK(cluster::hasEngine(st.caps));
    session.disconnect(true);
    setEnv("FAKE_CONTAINER_SITE", "");
}

TEST_CASE("cluster: a worker without the engine it was asked for fails at its hello, and is stopped", "[app][cluster][engine]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the worker");
    copyCheckout(fc.home / "sirius");
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    cluster::Profile p = engineProfile(fc);
    p.engineBin = pythonOnlyEngine(fc).generic_string();
    const cluster::Status st = connectUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    // not a connection: the job held, the Hello step failed, in words, with the fix
    CHECK(st.state == cluster::State::JobReady);
    CHECK(st.steps[static_cast<int>(cluster::Step::Hello)].status == cluster::StepStatus::Failed);
    CHECK(st.noEngine);
    CHECK(st.reason.find("is the Python worker alone: there is no SIRIUS C++ engine in this job") != std::string::npos);
    CHECK(st.fix.find("Set Engine builds folder") != std::string::npos);
    CHECK_FALSE(session.connected());
    // the worker step that came up without the engine was ended: it holds nothing
    CHECK(readAll(fc.slurm / "cancelled_steps").find("4711.") != std::string::npos);
    CHECK(cluster::wizard::hpcNoEngineReason(st) == "HPC: no SIRIUS engine in job 4711 \xE2\x80\x94 open Cluster to fix");
    const cluster::ConnectionBadge b = cluster::connectionBadge(st, false, std::chrono::steady_clock::now());
    CHECK(b.kind == cluster::ConnectionBadge::Kind::Failed);
    CHECK(b.label.find("no engine") != std::string::npos);
    session.disconnect(true);
    setEnv("FAKE_CONTAINER_SITE", "");
}

TEST_CASE("cluster: the launch script never starts the Python worker in place of a missing engine", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "w.sif") << "image\n";
    fs::create_directories(fc.home / ".sirius" / "run");
    writeAppScript(fc.home / ".sirius" / "run" / cluster::workerLaunchScriptName());
    ssh::Session s;
    s.open(fc.options(), {}, std::chrono::seconds(60));
    const std::string run = "bash \"$HOME/.sirius/run/" + cluster::workerLaunchScriptName() + "\"";
    const std::string base = "cd ~ && export SIRIUS_CONTAINER=\"$HOME/w.sif\" SIRIUS_TOKEN_FILE=\"$HOME/.sirius/run/token.x\" "
                             "SIRIUS_WORKER_DIR=\"$HOME/sirius/app/python\" SIRIUS_ENGINE=1 && ";
    // a build folder without its executable: said, exit 3, nothing started
    ssh::CommandResult r = s.run(base + "SIRIUS_ENGINE_DIR=\"$HOME/builds/abc\" " + run);
    INFO(r.out << "\n"
               << r.err);
    CHECK(r.exitCode == 3);
    CHECK(r.out.find("{\"error\": \"engine_missing\"") != std::string::npos);
    CHECK(r.err.find("SIRIUS's C++ engine is not at") != std::string::npos);
    CHECK(r.err.find("Engine builds folder") != std::string::npos);
    CHECK(readAll(fc.slurm / "apptainer.args").find("sirius_worker") == std::string::npos);
    // the image's own engine, which the image does not have (the fake runs `test -x` here)
    r = s.run(base + run);
    CHECK(r.exitCode == 3);
    CHECK(r.err.find("not in the image") != std::string::npos);
    CHECK(readAll(fc.slurm / "apptainer.args").find("serve") == std::string::npos);
    // a build that is there: its libraries go in front inside the image, its python/ is the worker's code by default
    const fs::path b = fc.home / "builds" / "abc";
    fs::create_directories(b / "bin");
    fs::create_directories(b / "lib");
    fs::create_directories(b / "python" / "sirius_worker");
    std::ofstream(b / "python" / "sirius_worker" / "__main__.py") << "\n";
    writeScript(b / "bin" / "sirius-cli", "#!/bin/bash\nexit 0\n");
    r = s.run("cd ~ && export FAKE_APPTAINER_DRY=1 SIRIUS_CONTAINER=\"$HOME/w.sif\" SIRIUS_TOKEN_FILE=\"$HOME/.sirius/run/token.x\" SIRIUS_ENGINE=1 "
              "SIRIUS_ENGINE_DIR=\"$HOME/builds/abc\" && " +
              run);
    INFO(r.out << "\n"
               << r.err);
    CHECK(r.ok());
    const std::string args = readAll(fc.slurm / "apptainer.args");
    INFO(args);
    CHECK(args.find("/bin/sh -c") != std::string::npos);
    CHECK(args.find("LD_LIBRARY_PATH=\"$d/lib") != std::string::npos);
    CHECK(args.find("builds/abc/bin/sirius-cli serve") != std::string::npos);
    CHECK(args.find("--worker-dir ") != std::string::npos);
    CHECK(args.find("builds/abc/python") != std::string::npos);
    s.close();
}

TEST_CASE("cluster: BUILD.json's schema hash under either key, the python column, the fixes", "[app][cluster]") {
    BuildInfo app = buildInfo();
    app.commit = "f5a2303000000000000000000000000000000000";
    // the cluster agent's BUILD.json: both keys, the same hash; or only one of them
    nlohmann::json both = toJson(app);
    both["commit"] = "8368aab4582f71273b16eaa63ce09d38ba867d8e";
    both["build"] = "0.1.0+g8368aab";
    both["schema_hash"] = app.opsSchema;
    nlohmann::json onlyNew = both;
    onlyNew.erase("ops_schema");
    nlohmann::json apiText = onlyNew;
    apiText.erase("api");
    apiText["engine_api"] = std::to_string(app.api);
    CHECK(buildInfoFromJson(onlyNew).opsSchema == app.opsSchema);
    CHECK(buildInfoFromJson(onlyNew).api == app.api);
    CHECK(buildInfoFromJson(apiText).api == app.api);
    CHECK(buildInfoFromJson(both).opsSchema == app.opsSchema);
    CHECK(engineMismatch(app, buildInfoFromJson(onlyNew)).empty());
    // the builds listing, with the python column and without (a script of before)
    const std::string out = "builds=yes\n@@build 8368aab yes yes " + onlyNew.dump() + "\n@@build old yes " + both.dump() + "\n@@build nopy yes no " +
                            apiText.dump() + "\n";
    const std::vector<cluster::EngineBuild> found = cluster::parseEngineBuilds(out);
    REQUIRE(found.size() == 3);
    CHECK(found[0].python);
    CHECK(found[0].readable);
    CHECK(found[0].info.opsSchema == app.opsSchema);
    CHECK_FALSE(found[1].python);
    CHECK(found[1].readable);
    CHECK_FALSE(found[2].python);
    CHECK(found[2].readable);
    std::string note;
    CHECK(cluster::pickEngineBuild(found, app, &note) == 0);
    CHECK(note == "the same operations as this build");
    // another schema: none fits, and the fix says to build this commit, with how
    nlohmann::json changed = onlyNew;
    changed["schema_hash"] = "0000";
    CHECK(cluster::pickEngineBuild(cluster::parseEngineBuilds("@@build 8368aab yes yes " + changed.dump() + "\n"), app, &note) == -1);
    CHECK(note.find("their operations or engine API differ") != std::string::npos);
    // the same operations, but no bin/sirius-cli: a broken install, said as such
    CHECK(cluster::pickEngineBuild(cluster::parseEngineBuilds("@@build 8368aab4582f no yes " + onlyNew.dump() + "\n"), app, &note) == -1);
    CHECK(note.find("8368aab4582f matches this application but has no bin/sirius-cli") != std::string::npos);
    // a bin/sirius-cli without an execute bit in its mode, or no lib/: the same, with what is wrong
    const std::vector<cluster::EngineBuild> nox = cluster::parseEngineBuilds("@@build 8368aab4582f nox yes yes " + onlyNew.dump() + "\n");
    REQUIRE(nox.size() == 1);
    CHECK(nox[0].binThere);
    CHECK_FALSE(nox[0].runnable);
    CHECK(nox[0].python);
    CHECK(nox[0].lib);
    CHECK(nox[0].readable);
    CHECK(cluster::pickEngineBuild(nox, app, &note) == -1);
    CHECK(note.find("without an execute bit in its mode (chmod +x bin/sirius-cli)") != std::string::npos);
    CHECK(cluster::pickEngineBuild(cluster::parseEngineBuilds("@@build 8368aab4582f yes yes no " + onlyNew.dump() + "\n"), app, &note) == -1);
    CHECK(note.find("has no lib/ folder") != std::string::npos);
    CHECK(cluster::pickEngineBuild(cluster::parseEngineBuilds("@@build 8368aab4582f yes yes yes " + onlyNew.dump() + "\n"), app, &note) == 0);
    CHECK(note.find("operations or engine API differ") == std::string::npos);
    const std::string fix = cluster::engineBuildFix("/clusterfs/me/sirius-builds", app);
    CHECK(fix.find("/clusterfs/me/sirius-builds/" + app.commit) != std::string::npos);
    CHECK(fix.find("build_sirius_engine.sbatch") != std::string::npos);
    CHECK(fix.find("README.md") != std::string::npos);
    CHECK(cluster::noEngineFix("").find("e.g. ~/sirius-builds") != std::string::npos);
    CHECK(cluster::engineBuildsScript("~/b").find("python/sirius_worker/__main__.py") != std::string::npos);
    // the login node is asked about files only: never whether it can run one (it may mount the builds noexec)
    const std::string listing = cluster::engineBuildsScript("~/b");
    CHECK(listing.find("ls -lLd") != std::string::npos);
    CHECK(listing.find("-x ") == std::string::npos);
    CHECK(listing.find("[ -d \"$EB/$d/lib\" ]") != std::string::npos);
    // the upload: base64 through the command channel, 0700, by the build's name
    const std::string up = cluster::uploadLaunchScript();
    CHECK(up.find("base64 -d") != std::string::npos);
    CHECK(up.find("chmod 700") != std::string::npos);
    CHECK(up.find(cluster::workerLaunchScriptName()) != std::string::npos);
    CHECK(up.find("SIRIUS_ENGINE") == std::string::npos);   // the text itself never meets the shell
}

// --- the checks run where the engine runs: in the job, on its node, in the image ----------------

TEST_CASE("cluster: the checks in the job, read: each line in words with its fix, the engine's version against BUILD.json", "[app][cluster]") {
    const BuildInfo app = buildInfo();
    cluster::Profile p;
    p.container = "~/w.sif";
    p.engine = true;
    p.engineBuilds = "~/b";
    p.bind = "~/data:/data, /scratch";
    p.gpus = 1;
    p.scratch = "/tmp/cache";
    nlohmann::json bj = toJson(app);
    bj["commit"] = "8368aab4582f71273b16eaa63ce09d38ba867d8e";
    bj["build"] = "0.1.0+g8368aab";
    const std::string dir = "~/b/8368aab4582f71273b16eaa63ce09d38ba867d8e";
    const auto envelope = [](const nlohmann::json& build, int cudaDevices) {
        return nlohmann::json{{"command", "version"},
                              {"ok", true},
                              {"result", {{"name", "sirius-cli"}, {"build", build}, {"features", {{"cuda", true}, {"cuda_devices", cudaDevices}}}}}};
    };
    const std::string head = "check launcher ok apptainer\ncheck image ok starts on g0004\ncheck python ok python 3.11, sirius, numpy\n"
                             "check torch warn not in the image: models will not run\ncheck worker ok /x/python\n";
    const std::string tail = "check gpu ok 1 visible (CUDA_VISIBLE_DEVICES=0)\ncheck data:0 ok /data\ncheck data:1 ok /scratch\ncheck cache ok /tmp/cache\n@@check-exit 0\n";

    // all well: no failure, each check in words, the engine's own build named
    cluster::NodeCheckReport rep = cluster::readNodeChecks(head + "@@version " + envelope(bj, 1).dump() + "\ncheck engine ok " + dir + "/bin/sirius-cli\n" + tail,
                                                           p, buildInfoFromJson(bj), dir, app, "g0004");
    CHECK(rep.failure() == nullptr);
    CHECK(rep.exitCode == 0);
    CHECK_FALSE(rep.noEngine);
    const auto find = [](const cluster::NodeCheckReport& r, const std::string& name) -> const cluster::NodeCheck* {
        for (const cluster::NodeCheck& c : r.checks)
            if (c.name == name) return &c;
        return nullptr;
    };
    REQUIRE(find(rep, "engine"));
    CHECK(find(rep, "engine")->status == cluster::StepStatus::Done);
    CHECK(find(rep, "engine")->detail.find("0.1.0+g8368aab (8368aab458) runs in the image, from " + dir) != std::string::npos);
    REQUIRE(find(rep, "cuda"));
    CHECK(find(rep, "cuda")->detail == "1 device for the engine");
    CHECK(find(rep, "torch")->status == cluster::StepStatus::Warning);
    CHECK_FALSE(find(rep, "torch")->fix.empty());
    CHECK(find(rep, "data:0")->detail == "~/data");   // as the profile names it
    CHECK(find(rep, "data:1")->detail == "/scratch");
    CHECK(find(rep, "cache")->detail == "/tmp/cache");
    CHECK(find(rep, "launcher")->detail == "apptainer");

    // the engine's CUDA finds no GPU of the step's: a warning, with what to do
    rep = cluster::readNodeChecks(head + "@@version " + envelope(bj, 0).dump() + "\ncheck engine ok x\n" + tail, p, buildInfoFromJson(bj), dir, app, "g0004");
    CHECK(rep.failure() == nullptr);
    CHECK(find(rep, "cuda")->status == cluster::StepStatus::Warning);

    // the folder holds another build's executable: refused, a broken install
    nlohmann::json other = bj;
    other["commit"] = "1111111111111111111111111111111111111111";
    other["build"] = "0.1.0+g1111111";
    rep = cluster::readNodeChecks(head + "@@version " + envelope(other, 1).dump() + "\ncheck engine ok x\n" + tail, p, buildInfoFromJson(bj), dir, app, "g0004");
    REQUIRE(rep.failure());
    CHECK(rep.failure()->name == "engine");
    CHECK(rep.failure()->detail.find("is 0.1.0+g1111111, but its BUILD.json says 0.1.0+g8368aab") != std::string::npos);
    CHECK(rep.failure()->fix.find("Reinstall that build") != std::string::npos);
    CHECK(rep.noEngine);

    // an executable that does not answer `version` as SIRIUS's engine does
    rep = cluster::readNodeChecks(head + "@@version {\"something\": 1}\ncheck engine ok x\n" + tail, p, buildInfoFromJson(bj), dir, app, "g0004");
    REQUIRE(rep.failure());
    CHECK(rep.failure()->detail.find("did not answer `version` as SIRIUS's engine does") != std::string::npos);

    // `version` fails: the node's own words kept for Details
    rep = cluster::readNodeChecks(head + "sirius-cli: error while loading shared libraries: libnvtiff.so.0\ncheck engine fail sirius-cli version exited with 127\n" + tail,
                                  p, buildInfoFromJson(bj), dir, app, "g0004");
    REQUIRE(rep.failure());
    CHECK(rep.failure()->detail.find("SIRIUS's engine does not run in the image on g0004") != std::string::npos);
    CHECK(rep.failure()->fix.find("lib/") != std::string::npos);
    CHECK(rep.output.find("libnvtiff.so.0") != std::string::npos);
    CHECK(rep.noEngine);

    // a data folder not on the node: named as the profile has it, before the image is tried
    rep = cluster::readNodeChecks("check launcher ok apptainer\ncheck data:1 fail /scratch is not on g0004\n", p, buildInfoFromJson(bj), dir, app, "g0004");
    REQUIRE(rep.failure());
    CHECK(rep.failure()->detail == "The data folder /scratch is not a folder on g0004: apptainer would stop before the worker starts.");
    CHECK(rep.failure()->fix.find("Data folders") != std::string::npos);

    // an image that does not start: no line from inside it
    rep = cluster::readNodeChecks("check launcher ok apptainer\nFATAL:   while loading image: could not mount squashfs\n@@check-exit 255\n", p,
                                  buildInfoFromJson(bj), dir, app, "g0004");
    REQUIRE(rep.failure());
    CHECK(rep.failure()->name == "image");
    CHECK(rep.failure()->detail.find("cannot run the worker on g0004: it does not start") != std::string::npos);
    CHECK(rep.output.find("squashfs") != std::string::npos);
    CHECK(rep.exitCode == 255);

    // nothing came back at all: the step did not run (srun's words kept)
    rep = cluster::readNodeChecks("srun: error: Unable to confirm allocation for job 4711\n", p, buildInfoFromJson(bj), dir, app, "g0004");
    REQUIRE(rep.failure());
    CHECK(rep.failure()->name == "step");
    CHECK(rep.output.find("Unable to confirm allocation") != std::string::npos);

    // the image's own engine is not there: no engine, with the fix that names a builds folder
    cluster::Profile own = p;
    own.engineBuilds.clear();
    rep = cluster::readNodeChecks(head + "check engine-missing fail /opt/sirius/bin/sirius-cli is not there inside the image\n" + tail, own, std::nullopt, {}, app,
                                  "g0004", "/clusterfs/me/sirius-builds");
    REQUIRE(rep.failure());
    CHECK(rep.noEngine);
    CHECK(rep.failure()->detail.rfind("No SIRIUS C++ engine found: the image ~/w.sif has no /opt/sirius/bin/sirius-cli", 0) == 0);
    CHECK(rep.failure()->fix.find("e.g. /clusterfs/me/sirius-builds") != std::string::npos);

    // the data folders go to the check as host / in-image pairs
    const std::vector<std::pair<std::string, std::string>> pairs = cluster::bindPairs("~/data:/data:ro, /scratch ,,");
    REQUIRE(pairs.size() == 2);
    CHECK(pairs[0] == std::make_pair(std::string("~/data"), std::string("/data")));
    CHECK(pairs[1] == std::make_pair(std::string("/scratch"), std::string("/scratch")));
}

TEST_CASE("cluster: a build on scratch the login node mounts noexec is listed by its mode and checked where it runs", "[app][cluster][engine]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the checks");
    copyCheckout(fc.home / "sirius");
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    // the builds on a scratch file system the login node mounts noexec (fiona's /clusterfs/nvme2)
    const std::string commit = "8368aab4582f71273b16eaa63ce09d38ba867d8e";
    nlohmann::json bj = toJson(buildInfo());
    bj["commit"] = commit;
    bj["build"] = "0.1.0+g8368aab";
    const fs::path builds = fc.root / "noexec-nvme2" / "sirius-builds";
    const fs::path b = engineBuild(builds, commit, bj);
    setEnv("FAKE_SSH_NOEXEC", "noexec-nvme2");
    // the login node says access(X_OK) "no" for the build's executable, though its mode has the bit
    {
        ssh::Session s;
        s.open(fc.options(), {}, std::chrono::seconds(60));
        const std::string f = (b / "bin" / "sirius-cli").generic_string();
        const ssh::CommandResult r = s.run("f='" + f + "'; [ -x \"$f\" ] && echo x=yes || echo x=no; test -x \"$f\" && echo t=yes || echo t=no; "
                                                       "[ -f \"$f\" ] && echo f=yes; ls -lLd \"$f\" | cut -c1-10");
        INFO(r.out << r.err);
        CHECK(r.out.find("x=no") != std::string::npos);
        CHECK(r.out.find("t=no") != std::string::npos);
        CHECK(r.out.find("f=yes") != std::string::npos);
        CHECK(r.out.find("-rwx") != std::string::npos);
        s.close();
    }
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    cluster::Profile p = engineProfile(fc);
    p.engineBin.clear();
    p.engineBuilds = builds.generic_string();
    const cluster::Status st = connectUntilSettled(session, p);
    setEnv("FAKE_SSH_NOEXEC", "");
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    // accepted: listed by its mode on the login node, run in the job on the node
    REQUIRE(st.state == cluster::State::Connected);
    CHECK(st.engineBuild.find(commit) != std::string::npos);
    CHECK(cluster::hasEngine(st.caps));
    const cluster::NodeCheck* engine = nullptr;
    for (const cluster::NodeCheck& c : st.nodeChecks)
        if (c.name == "engine") engine = &c;
    REQUIRE(engine);
    CHECK(engine->status == cluster::StepStatus::Done);
    CHECK(engine->detail.find("0.1.0+g8368aab") != std::string::npos);
    // the check ran as a step of the job, through srun with its id, before the worker
    const std::string steps = readAll(fc.slurm / "srun.args");
    CHECK(steps.find("--jobid=4711 --overlap --nodes=1 --ntasks=1 --job-name=sirius-check bash ") != std::string::npos);
    CHECK(steps.find(" --check") != std::string::npos);
    CHECK(steps.find("--job-name=sirius-check") < steps.find("--job-name=sirius-worker"));
    CHECK(readAll(fc.slurm / "4711.checks").find(cluster::workerLaunchScriptName() + " --check") != std::string::npos);
    // nothing moved or linked: the build stays where it is
    CHECK(fs::is_regular_file(b / "bin" / "sirius-cli"));
    CHECK_FALSE(fs::is_symlink(b / "bin" / "sirius-cli"));
    session.disconnect(true);
    setEnv("FAKE_CONTAINER_SITE", "");
}

TEST_CASE("cluster: an engine build whose `version` fails in the job is refused with its output, and nothing starts", "[app][cluster][engine]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the checks");
    copyCheckout(fc.home / "sirius");
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    const fs::path b = engineBuild(fc.home / "sirius-builds", "deadbeef", toJson(buildInfo()));
    // its executable is there, with its execute bit, and the login node lists it: it does not run on the node
    writeScript(b / "bin" / "sirius-cli",
                "#!/bin/bash\necho 'sirius-cli: error while loading shared libraries: libnvtiff.so.0: cannot open shared object file: No such file or directory' >&2\n"
                "exit 127\n");
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    cluster::Profile p = engineProfile(fc);
    p.engineBin.clear();
    p.engineBuilds = "~/sirius-builds";
    const cluster::Status st = connectUntilSettled(session, p);
    setEnv("FAKE_CONTAINER_SITE", "");
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    CHECK(st.state == cluster::State::JobReady);   // the job holds on
    CHECK(st.steps[static_cast<int>(cluster::Step::Checks)].status == cluster::StepStatus::Failed);
    CHECK(st.reason.find("SIRIUS's engine does not run in the image on fakenode") != std::string::npos);
    CHECK(st.reason.find("version exited with 127") != std::string::npos);
    CHECK(st.remoteOutput.find("libnvtiff.so.0") != std::string::npos);
    CHECK(st.fix.find("lib/") != std::string::npos);
    CHECK(st.noEngine);
    CHECK(noWorkerStarted(fc));   // never the Python worker instead, never a worker at all
    CHECK(readAll(fc.slurm / "srun.args").find("--jobid=4711 --overlap --nodes=1 --ntasks=1 --job-name=sirius-check") != std::string::npos);
    // the Worker page's checklist: the engine's line failed, the others as found
    const cluster::wizard::HealthReport r = cluster::wizard::healthReport(st, p, buildInfo(), std::chrono::steady_clock::now());
    CHECK(r.verdict == cluster::wizard::HealthReport::Verdict::Failed);
    bool engineRow = false, pythonRow = false;
    for (const cluster::wizard::HealthRow& row : r.rows) {
        if (row.label == "C++ engine" && row.mark == cluster::wizard::Mark::Fail) engineRow = true;
        if (row.label == "Python" && row.mark == cluster::wizard::Mark::Ok) pythonRow = true;
    }
    CHECK(engineRow);
    CHECK(pythonRow);
    session.disconnect(true);
}

TEST_CASE("cluster: the checks and the worker ask for the job's GPUs; a step given more is held to them", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the worker");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "w.sif") << "image\n";
    setEnv("FAKE_CONTAINER_SITE", containerSite(fc).string());
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    fc.fakeNvidiaSmi({"GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-a)", "GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-b)",
                      "GPU 2: NVIDIA A100-SXM4-80GB (UUID: GPU-c)", "GPU 3: NVIDIA A100-SXM4-80GB (UUID: GPU-d)"});
    // a node that hands every step all four of its GPUs, whatever the step asked for
    std::ofstream(fc.slurm / "step_gpus") << "0,1,2,3\n";
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    cluster::Profile p = containerProfile(fc, "~/w.sif");
    p.partition = "dgx";   // a partition with GPUs (the default one has none)
    p.gpus = 1;
    cluster::Status st = connectUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    const std::string steps = readAll(fc.slurm / "srun.args");
    CHECK(steps.find("--job-name=sirius-check --gres=gpu:1 ") != std::string::npos);
    CHECK(steps.find("--job-name=sirius-worker --gres=gpu:1 ") != std::string::npos);
    CHECK(readAll(fc.slurm / "4711.gpus").find("sirius-check 0,1,2,3") != std::string::npos);
    const auto gpuCheck = [](const cluster::Status& s) {
        for (const cluster::NodeCheck& c : s.nodeChecks)
            if (c.name == "gpu") return c;
        return cluster::NodeCheck{};
    };
    cluster::NodeCheck g = gpuCheck(st);
    CHECK(g.status == cluster::StepStatus::Done);
    CHECK(g.detail == "1 visible (CUDA_VISIBLE_DEVICES=0); the step was given 4 GPUs (0,1,2,3), the job asked for 1: held to 0");
    CHECK(st.steps[static_cast<int>(cluster::Step::Checks)].detail.find("GPUs: 1 visible") != std::string::npos);
    // a node that confines the step to what it asked for: no note
    fs::remove(fc.slurm / "step_gpus");
    st = workerUntilSettled(session, p);
    REQUIRE(st.state == cluster::State::Connected);
    g = gpuCheck(st);
    CHECK(g.status == cluster::StepStatus::Done);
    CHECK(g.detail == "1 visible (CUDA_VISIBLE_DEVICES=0)");
    session.disconnect(true);
    setEnv("FAKE_CONTAINER_SITE", "");

    // the worker's own start: the step's CUDA_VISIBLE_DEVICES held to the job's GPUs, handed into the image
    ssh::Session s;
    s.open(fc.options(), {}, std::chrono::seconds(60));
    fs::remove(fc.slurm / "apptainer.args");
    const std::string job = "cd ~/sirius && mkdir -p ~/.sirius/run && export FAKE_APPTAINER_DRY=1 SIRIUS_CONTAINER=\"$HOME/w.sif\" "
                            "SIRIUS_TOKEN_FILE=\"$HOME/.sirius/run/token.x\" SLURM_SUBMIT_DIR=\"$PWD\" CUDA_VISIBLE_DEVICES=0,1,2,3 SIRIUS_GPUS=1 && ";
    const ssh::CommandResult r = s.run(job + "bash app/python/slurm/sirius_worker.sbatch");
    INFO(r.out << "\n"
               << r.err);
    CHECK(r.ok());
    CHECK(r.out.find("GPUs: CUDA_VISIBLE_DEVICES=0 (the step was given 4 GPUs (0,1,2,3), the job asked for 1: held to 0)") != std::string::npos);
    const std::string args = readAll(fc.slurm / "apptainer.args");
    const std::size_t at = args.find("--env-file ");
    REQUIRE(at != std::string::npos);
    const std::string envFile = args.substr(at + 11, args.find(' ', at + 11) - (at + 11));
    const std::string env = s.run("cat '" + envFile + "'").out;   // the cluster's path for it
    INFO(envFile << "\n"
                 << env);
    CHECK(env.find("CUDA_VISIBLE_DEVICES=0") != std::string::npos);
    CHECK(env.find("CUDA_VISIBLE_DEVICES=0,") == std::string::npos);
    s.close();
}

// --- a job of the user's own --------------------------------------------------------------------

TEST_CASE("cluster: the user's jobs as squeue lists them, their GPUs and a line for each", "[app][cluster]") {
    CHECK(cluster::gresGpus("gpu:2") == 2);
    CHECK(cluster::gresGpus("gres:gpu:a100:2") == 2);
    CHECK(cluster::gresGpus("gres/gpu:a100=2") == 2);
    CHECK(cluster::gresGpus("gres/gpu=4") == 4);
    CHECK(cluster::gresGpus("gres/gpu") == 1);
    CHECK(cluster::gresGpus("gres/gpu:a100=2,gres/gpu:v100=1") == 3);
    CHECK(cluster::gresGpus("N/A") == 0);
    CHECK(cluster::gresGpus("(null)") == 0);
    CHECK(cluster::gresGpus("") == 0);
    const std::vector<cluster::ClusterJob> jobs = cluster::parseUserJobs("5001|jupyter|gpu-a100|RUNNING|1:02:03|2-00:00:00|1|g0004|gres/gpu:a100=2|16|128G\n"
                                                                         "5002|train|cpu|PENDING|0:00|1:00:00|1|(Resources)|N/A|4|8G\n"
                                                                         "squeue: error: something to ignore\n\n");
    REQUIRE(jobs.size() == 2);
    CHECK(jobs[0].id == "5001");
    CHECK(jobs[0].name == "jupyter");
    CHECK(jobs[0].partition == "gpu-a100");
    CHECK(jobs[0].running());
    CHECK(jobs[0].gpus == 2);
    CHECK(jobs[0].cpus == 16);
    CHECK(jobs[0].mem == "128G");
    CHECK(jobs[0].where == "g0004");
    CHECK(cluster::jobSummary(jobs[0]) == "g0004 \xC2\xB7 2 GPUs \xC2\xB7 16 CPUs \xC2\xB7 128G \xC2\xB7 1:02:03 of 2-00:00:00");
    CHECK_FALSE(jobs[1].running());
    CHECK(jobs[1].gpus == 0);
    CHECK(cluster::jobSummary(jobs[1]) == "pending (Resources) \xC2\xB7 no GPU \xC2\xB7 4 CPUs \xC2\xB7 8G \xC2\xB7 time limit 1:00:00");
    CHECK(cluster::userJobsScript().find("squeue -h -u") != std::string::npos);
    CHECK(cluster::userJobsScript().find("RUNNING,PENDING") != std::string::npos);
}

namespace {
    // A job of the user's own on the fake cluster, not SIRIUS's: running (or
    // waiting, `pendingPolls` > 0), `info` = "name|partition|gres|cpus|mem|node".
    void ownJob(const FakeCluster& fc, const std::string& id, const std::string& info, int pendingPolls = 0) {
        std::ofstream(fc.slurm / (id + ".state")) << "RUNNING\n";
        std::ofstream(fc.slurm / (id + ".holder")) << "\n";
        std::ofstream(fc.slurm / (id + ".pending")) << pendingPolls << "\n";
        std::ofstream(fc.slurm / (id + ".info")) << info << "\n";
    }
} // namespace

TEST_CASE("cluster: one of the user's own jobs is taken up, its worker runs in it, and SIRIUS never cancels it", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the worker");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "w.sif") << "image\n";
    setEnv("FAKE_CONTAINER_SITE", containerSite(fc).string());
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    ownJob(fc, "5001", "jupyter|gpu-a100|gres/gpu:a100=2|16|128G|fakenode");
    ownJob(fc, "5002", "train|cpu|N/A|4|8G|", 1000);
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    const cluster::Profile p = containerProfile(fc, "~/w.sif");

    // logged in: the user's jobs, SIRIUS's or not, running and waiting
    session.logIn(p);
    REQUIRE(waitFor([&] { return settled(session) && session.status().sshUp; }, std::chrono::seconds(120)));
    std::vector<cluster::ClusterJob> jobs = session.listJobs();
    REQUIRE(jobs.size() == 2);
    CHECK(jobs[0].id == "5001");
    CHECK(jobs[0].gpus == 2);
    CHECK(jobs[0].running());
    CHECK(jobs[1].id == "5002");
    CHECK(jobs[1].state == "PENDING");

    // take up 5001: no sbatch, the job's own GPUs for the checks and the worker
    session.adoptJob(p, "5001");
    REQUIRE(waitFor([&] { return settled(session); }, std::chrono::seconds(120)));
    cluster::Status st = session.status();
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::JobReady);
    CHECK(st.jobId == "5001");
    CHECK(st.adopted);
    CHECK_FALSE(fs::exists(fc.slurm / "4711.args"));   // nothing submitted
    CHECK(st.steps[static_cast<int>(cluster::Step::Submit)].detail.find("job 5001 (yours)") != std::string::npos);
    CHECK(cluster::wizard::jobLine(st, std::chrono::steady_clock::now()).text.find("never cancelled by SIRIUS") != std::string::npos);
    st = workerUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    const std::string steps = readAll(fc.slurm / "srun.args");
    CHECK(steps.find("--jobid=5001 --overlap --nodes=1 --ntasks=1 --job-name=sirius-check --gres=gpu:2 ") != std::string::npos);
    CHECK(steps.find("--jobid=5001 --overlap --nodes=1 --ntasks=1 --job-name=sirius-worker --gres=gpu:2 ") != std::string::npos);

    // disconnect, even asked to cancel: the job keeps running, and says why
    session.disconnect(true);
    st = session.status();
    CHECK(st.reason.find("job 5001 was yours before SIRIUS: it keeps running") != std::string::npos);
    CHECK(readAll(fc.slurm / "cancelled").find("5001") == std::string::npos);
    CHECK(readAll(fc.slurm / "5001.state").rfind("RUNNING", 0) == 0);

    // connect again: taken up again, still the user's
    session.connectJob(p);
    REQUIRE(waitFor([&] { return settled(session); }, std::chrono::seconds(120)));
    st = session.status();
    INFO(st.reason);
    CHECK((st.state == cluster::State::JobReady || st.state == cluster::State::Connected));
    CHECK(st.jobId == "5001");
    CHECK(st.adopted);
    CHECK_FALSE(fs::exists(fc.slurm / "4711.args"));

    // Let go of it (Change job): not cancelled, the login kept
    session.cancelJob();
    st = session.status();
    CHECK(st.state == cluster::State::Idle);
    CHECK(st.jobId.empty());
    CHECK_FALSE(st.adopted);
    CHECK(st.reason.find("was yours before SIRIUS: it keeps running") != std::string::npos);
    CHECK(readAll(fc.slurm / "cancelled").find("5001") == std::string::npos);
    CHECK(readAll(fc.slurm / "5001.state").rfind("RUNNING", 0) == 0);

    // a job that is not the user's (or not there): said, nothing taken up
    session.adoptJob(p, "9999");
    REQUIRE(waitFor([&] { return settled(session); }, std::chrono::seconds(120)));
    st = session.status();
    CHECK(st.steps[static_cast<int>(cluster::Step::Submit)].status == cluster::StepStatus::Failed);
    CHECK(st.reason.find("Job 9999 is not among your running or waiting jobs") != std::string::npos);
    CHECK(st.jobId.empty());
    session.disconnect(true);
    CHECK(readAll(fc.slurm / "cancelled").find("5001") == std::string::npos);
    setEnv("FAKE_CONTAINER_SITE", "");
}

TEST_CASE("cluster: an array job id is not globbed by the login shell", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    const std::string id = "12345_[1-10]";
    ownJob(fc, id, "jupyter|gpu-a100|gres/gpu:a100=1|4|8G|fakenode");
    std::ofstream(fc.home / "12345_1") << "decoy\n";   // matches the unquoted glob 12345_[1-10]
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    const cluster::Profile p = containerProfile(fc, "");
    session.logIn(p);
    REQUIRE(waitFor([&] { return settled(session) && session.status().sshUp; }, std::chrono::seconds(120)));
    session.adoptJob(p, id);
    REQUIRE(waitFor([&] { return settled(session); }, std::chrono::seconds(120)));
    const cluster::Status st = session.status();
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    CHECK(st.jobId == id);
    // An unquoted squeue -j globs the decoy file and the job never becomes ready.
    CHECK((st.state == cluster::State::JobReady || st.state == cluster::State::Connected));
    CHECK(st.steps[static_cast<int>(cluster::Step::Queue)].status != cluster::StepStatus::Failed);
    session.disconnect(false);
}

// --- the engine's Python child comes up after the hello ----------------------------------------

TEST_CASE("cluster wizard: the torch row follows the engine's Python child: starting, ready (with its CUDA), failed (with its last words)", "[app][cluster]") {
    namespace wz = cluster::wizard;
    const auto now = std::chrono::steady_clock::now();
    const BuildInfo app = buildInfo();
    cluster::Profile p;
    p.gpus = 1;
    p.engine = true;
    cluster::Status st;
    st.state = cluster::State::Connected;
    st.jobId = "4711";
    st.node = "g0004";
    nlohmann::json engine = toJson(app);
    engine["python"] = {{"state", "starting"}};
    st.caps.engine = engine;
    const auto torchRow = [&](const cluster::Status& s) {
        for (const wz::HealthRow& r : wz::healthReport(s, p, app, now).rows)
            if (r.label == "torch") return r;
        return wz::HealthRow{};
    };
    CHECK(torchRow(st).value.find("still starting") != std::string::npos);
    CHECK(torchRow(st).mark == wz::Mark::Info);
    // ready: its version, and whether it computes on the GPU
    engine["python"] = {{"state", "ready"}, {"caps", {{"torch", "2.5.1+cu124"}, {"cuda_usable", false}, {"cuda_reason", "torch 2.5.1 is a CPU-only build"}}}};
    st.caps.engine = engine;
    st.caps.torch = "2.5.1+cu124";
    CHECK(torchRow(st).value == "2.5.1+cu124 \xC2\xB7 CPU only: torch 2.5.1 is a CPU-only build");
    CHECK(torchRow(st).mark == wz::Mark::Ok);
    // failed: a failure, with the end of what it wrote
    engine["python"] = {{"state", "failed"}, {"error", "the Python worker did not start (exit code 1)"}, {"stderr", "Traceback (most recent call last):\nImportError: libcudart.so.12"}};
    st.caps.engine = engine;
    st.caps.torch.clear();
    const wz::HealthRow failed = torchRow(st);
    CHECK(failed.mark == wz::Mark::Fail);
    CHECK(failed.value.find("did not start (exit code 1)") != std::string::npos);
    CHECK(failed.value.find("ImportError: libcudart.so.12") != std::string::npos);
    // the checks in the job, once connected: their warnings in one row
    st.nodeChecks = {cluster::NodeCheck{"gpu", "GPUs", cluster::StepStatus::Warning, "Slurm named no GPU for this step", "ask"},
                     cluster::NodeCheck{"python", "Python", cluster::StepStatus::Done, "python 3.11, sirius, numpy", {}}};
    bool checksRow = false;
    for (const wz::HealthRow& r : wz::healthReport(st, p, app, now).rows)
        if (r.label == "Checks in the job") checksRow = r.mark == wz::Mark::Warn && r.value.find("GPUs: Slurm named no GPU") != std::string::npos;
    CHECK(checksRow);
}

TEST_CASE("cluster: the session asks the engine again until its Python child is up, and the report follows", "[app][cluster][engine]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the worker");
    copyCheckout(fc.home / "sirius");
    setEnv("FAKE_SLURM_PENDING_POLLS", "0");
    setEnv("FAKE_ENGINE_PYTHON", "1");   // the engine starts its Python child, as on a cluster
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(300));
    const cluster::Profile p = engineProfile(fc);
    cluster::Status st = connectUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    REQUIRE(cluster::hasEngine(st.caps));
    CHECK(std::find(st.caps.methods.begin(), st.caps.methods.end(), "capabilities") != st.caps.methods.end());
    const auto pythonState = [](const cluster::Status& s) {
        const nlohmann::json py = s.caps.engine.value("python", nlohmann::json::object());
        return py.is_object() ? py.value("state", std::string()) : std::string();
    };
    const std::string first = pythonState(st);
    const int serial = st.capsSerial;
    CHECK(serial >= 1);
    // the child is up (or has failed) in a while: the session hears of it without a new hello
    REQUIRE(waitFor([&] { return pythonState(session.status()) != "starting"; }, std::chrono::seconds(120)));
    st = session.status();
    INFO(st.caps.engine.dump());
    CHECK(pythonState(st) == "ready");
    if (first == "starting") CHECK(st.capsSerial > serial);
    // the report's torch row no longer says it is starting
    for (const cluster::wizard::HealthRow& r : cluster::wizard::healthReport(st, p, buildInfo(), std::chrono::steady_clock::now()).rows)
        if (r.label == "torch") CHECK(r.value.find("still starting") == std::string::npos);
    session.disconnect(true);
    setEnv("FAKE_ENGINE_PYTHON", "0");
    setEnv("FAKE_CONTAINER_SITE", "");
}
