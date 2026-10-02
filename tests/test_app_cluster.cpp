// The cluster connection (core/remote_host, core/cluster, core/remote_source)
// against a local stand-in: tests/tools/fake_ssh.py plays OpenSSH's ssh
// (prompts through SSH_ASKPASS, a SOCKS5 proxy on -D, `bash -s` as the
// remote shell), tests/tools/fake_slurm plays sbatch / squeue / sacct /
// scancel / sinfo / sacctmgr / scontrol / apptainer and starts the real
// worker (app/python) on 127.0.0.1. No real ssh is run and no host but this
// one is contacted.
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

#include <sirius/tiff_io.hpp>

#include "core/build_info.hpp"
#include "core/cluster.hpp"
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
            for (const char* tool : {"sbatch", "squeue", "sacct", "scancel", "sinfo", "sacctmgr", "scontrol", "apptainer"}) {
                std::string text = readAll(fs::path(SIRIUS_TEST_FAKE_SLURM_DIR) / tool);
                std::string lf;
                for (char c : text)
                    if (c != '\r') lf.push_back(c);
                writeScript(bin / tool, lf);
            }
            writeScript(bin / "python3", "#!/bin/bash\nexec \"$FAKE_PYTHON\" \"$@\"\n");
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
    cluster::Profile p;
    p.host = "fakecluster";
    p.sshProgram = fc.python;
    p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
    p.checkout = "~/sirius";
    p.venv = "";
    p.port = ssh::freeLocalPort();
    bool sawPending = false;
    session.setChanged([&] {
        const cluster::Status st = session.status();
        if (st.steps[static_cast<int>(cluster::Step::Queue)].detail.find("PENDING") != std::string::npos) sawPending = true;
    });
    session.connect(p);
    REQUIRE(waitFor([&] { return session.status().state != cluster::State::Connecting; }, std::chrono::seconds(180)));
    cluster::Status st = session.status();
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    CHECK(sawPending);
    CHECK(st.node == "fakenode");
    CHECK(st.jobId == "4711");
    CHECK(st.caps.protocolVersion == rpc::kProtocolVersion);
    CHECK_FALSE(st.caps.version.empty());
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
    CHECK(fs::exists(run / "sirius-worker-4711.log"));
    for (const auto& entry : fs::directory_iterator(run)) CHECK(entry.path().filename().string().rfind("token.", 0) != 0);
    CHECK(readAll(run / "sirius-worker-4711.log").find(session.endpoint().token) == std::string::npos);
#ifndef _WIN32
    CHECK((fs::status(run).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);
    CHECK(readAll(fc.slurm / "4711.tokenmode").rfind("600", 0) == 0);
    CHECK(readAll(fc.slurm / "4711.umask").rfind("0077", 0) == 0);
#endif
    CHECK(session.endpoint().port > 0);
    CHECK(args.find("--parsable") != std::string::npos);
    CHECK(args.find("--partition=abc_a100") != std::string::npos);
    CHECK(args.find(session.endpoint().token) == std::string::npos);
    CHECK(fc.sshLog().find(session.endpoint().token) == std::string::npos);

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
    cluster::Profile p;
    p.host = "fakecluster";
    p.sshProgram = fc.python;
    p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
    p.checkout = "~/sirius";
    p.venv = "";
    p.port = ssh::freeLocalPort();
    session.connect(p);
    REQUIRE(waitFor([&] { return session.status().state != cluster::State::Connecting; }, std::chrono::seconds(180)));
    INFO(session.status().reason);
    REQUIRE(session.connected());
    // the job ends behind the application's back (a wall-time limit, an admin)
    REQUIRE(session.run("scancel 4711").ok());
    REQUIRE(waitFor([&] { return !session.connected(); }, std::chrono::seconds(60)));
    const cluster::Status st = session.status();
    CHECK(st.reason.find("4711") != std::string::npos);
    CHECK(st.reason.find("CANCELLED") != std::string::npos);
    CHECK(st.sshUp);   // the login is kept: browsing goes on, Connect submits anew
    session.disconnect(false);
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

    // A stand-in for the image's packages: a sirius that imports.
    fs::path containerSite(const FakeCluster& fc) {
        const fs::path site = fc.root / "image-site";
        fs::create_directories(site / "sirius");
        std::ofstream(site / "sirius" / "__init__.py") << "__version__ = '0-test'\n";
        return site;
    }

    cluster::Profile containerProfile(const FakeCluster& fc, const std::string& image) {
        cluster::Profile p;
        p.host = "fakecluster";
        p.sshProgram = fc.python;
        p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
        p.checkout = "~/sirius";
        p.venv = "~/no-such-venv";   // ignored with a container
        p.container = image;
        p.port = ssh::freeLocalPort();
        return p;
    }

    cluster::Status connectUntilSettled(cluster::Session& session, const cluster::Profile& p) {
        session.connect(p);
        waitFor([&] { return session.status().state != cluster::State::Connecting; }, std::chrono::seconds(180));
        return session.status();
    }
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
    CHECK(w.find("only the image and your home folder") != std::string::npos);
    CHECK(w.find("add them under Bind") != std::string::npos);
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
    CHECK(m.find("add it under Bind") != std::string::npos);
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
    auto checksFailed = [](const cluster::Status& st) {
        return st.state == cluster::State::Disconnected && st.steps[static_cast<int>(cluster::Step::Checks)].status == cluster::StepStatus::Failed;
    };

    // no image there
    cluster::Status st = connectUntilSettled(session, containerProfile(fc, "~/missing.sif"));
    INFO(st.reason);
    CHECK(checksFailed(st));
    CHECK(st.reason.find("no container image at ~/missing.sif") != std::string::npos);
    CHECK(st.fix.find("Build the SIRIUS worker image") != std::string::npos);
    CHECK(st.fix.find("pip") == std::string::npos);

    // an image that does not open
    st = connectUntilSettled(session, containerProfile(fc, "~/broken.sif"));
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
    st = connectUntilSettled(session, containerProfile(fc, "~/empty.sif"));
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
    st = connectUntilSettled(session, containerProfile(fc, "~/empty.sif"));
    CHECK(checksFailed(st));
    CHECK(st.reason.find("Neither apptainer nor singularity") != std::string::npos);
    CHECK_FALSE(fs::exists(fc.slurm / "next"));   // nothing was submitted
    session.disconnect(false);
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
    // the job was told the image and the launcher, not the venv
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

TEST_CASE("cluster: the checks fail on a bind path that is not there, before anything is submitted", "[app][cluster]") {
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
    CHECK(st.state == cluster::State::Disconnected);
    CHECK(st.steps[static_cast<int>(cluster::Step::Checks)].status == cluster::StepStatus::Failed);
    CHECK(st.reason.find("~/missing") != std::string::npos);
    CHECK(st.reason.find("~/data") == std::string::npos);   // the one that is there is not named
    CHECK(st.fix.find("Bind") != std::string::npos);
    CHECK_FALSE(fs::exists(fc.slurm / "next"));   // nothing was submitted
    session.disconnect(false);
}

TEST_CASE("cluster: no bind is a warning; the binds and the Python path reach the job", "[app][cluster]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    if (!pythonHas(fc.python, "numpy")) SKIP("no numpy in " + fc.python + " for the image check");
    copyCheckout(fc.home / "sirius");
    std::ofstream(fc.home / "w.sif") << "image\n";
    fs::create_directories(fc.home / "data");
    setEnv("FAKE_CONTAINER_SITE", containerSite(fc).string());
    // sbatch refuses every job once it has recorded its environment: no worker is started
    std::ofstream(fc.slurm / "sbatch.fail") << "1\n";
    cluster::Session session;
    session.setPollInterval(std::chrono::milliseconds(200), std::chrono::milliseconds(500));
    cluster::Profile p = containerProfile(fc, "~/w.sif");

    // no bind: the checks pass with a warning, and the job is submitted all the same
    cluster::Status st = connectUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    const cluster::StepState checks = st.steps[static_cast<int>(cluster::Step::Checks)];
    CHECK(checks.status == cluster::StepStatus::Warning);
    CHECK(checks.detail.find("only the image and your home folder") != std::string::npos);
    CHECK(checks.detail.find("add them under Bind") != std::string::npos);
    CHECK(st.steps[static_cast<int>(cluster::Step::Submit)].status == cluster::StepStatus::Failed);
    CHECK_FALSE(st.home.empty());   // the checks said where $HOME is
    std::string env = readAll(fc.slurm / "4711.env");
    CHECK(env.find("SIRIUS_CONTAINER=") != std::string::npos);
    CHECK(env.find("SIRIUS_CONTAINER_BIND") == std::string::npos);
    CHECK(env.find("SIRIUS_CONTAINER_PYTHONPATH") == std::string::npos);

    // binds ("~/" made $HOME) and a Python path: checked, then in the job's environment
    fs::remove(fc.slurm / "apptainer.args");
    p.bind = "~/data:/data, /tmp";
    p.containerPythonPath = "/opt/extra";
    st = connectUntilSettled(session, p);
    const cluster::StepState checks2 = st.steps[static_cast<int>(cluster::Step::Checks)];
    INFO(checks2.detail);
    CHECK(checks2.status == cluster::StepStatus::Done);
    CHECK(checks2.detail.find("binds 2") != std::string::npos);
    env = readAll(fc.slurm / "4712.env");
    INFO(env);
    CHECK(env.find("SIRIUS_CONTAINER_BIND=") != std::string::npos);
    CHECK(env.find("/data:/data,/tmp\n") != std::string::npos);
    CHECK(env.find("SIRIUS_CONTAINER_BIND=~") == std::string::npos);
    CHECK(env.find("SIRIUS_CONTAINER_PYTHONPATH=/opt/extra\n") != std::string::npos);
    CHECK(env.find("SIRIUS_TOKEN=") == std::string::npos);   // never the token itself
    // the image was tried with the same binds
    CHECK(readAll(fc.slurm / "apptainer.args").find("--bind ") != std::string::npos);
    CHECK(readAll(fc.slurm / "apptainer.args").find("/data:/data,/tmp") != std::string::npos);
    session.disconnect(false);
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
        cluster::Profile p;
        p.host = "fakecluster";
        p.sshProgram = fc.python;
        p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
        p.checkout = "~/sirius";
        p.venv = "";
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

TEST_CASE("cluster: the profile keeps the engine, on by default with a container image", "[app][cluster]") {
    cluster::Profile p;
    CHECK_FALSE(p.engine);
    p.engine = true;
    p.engineBin = "/opt/sirius/bin/sirius-cli";
    const cluster::Profile back = cluster::Profile::fromJson(p.toJson());
    CHECK(back.engine);
    CHECK(back.engineBin == "/opt/sirius/bin/sirius-cli");
    CHECK(cluster::Profile::fromJson(nlohmann::json{{"container", "~/w.sif"}}).engine);
    CHECK_FALSE(cluster::Profile::fromJson(nlohmann::json{{"container", ""}}).engine);
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
    st = connectUntilSettled(session, p);
    INFO(st.reason << "\n"
                   << st.remoteOutput);
    REQUIRE(st.state == cluster::State::Connected);
    CHECK(st.jobId == "4711");
    CHECK_FALSE(fs::exists(fc.slurm / "4712.args"));
    CHECK(st.steps[static_cast<int>(cluster::Step::Submit)].detail.find("reattached") != std::string::npos);
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
    INFO(st.reason);
    CHECK(st.state == cluster::State::Disconnected);
    CHECK(st.steps[static_cast<int>(cluster::Step::Hello)].status == cluster::StepStatus::Failed);
    CHECK(st.reason.find("0.0.9+gdeadbee") != std::string::npos);
    CHECK(st.reason.find("their operations differ") != std::string::npos);
    session.disconnect(true);
}
