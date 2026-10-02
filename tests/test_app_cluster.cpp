// The cluster connection (core/remote_host, core/cluster, core/remote_source)
// against a local stand-in: tests/tools/fake_ssh.py plays OpenSSH's ssh
// (prompts through SSH_ASKPASS, a SOCKS5 proxy on -D, `bash -s` as the
// remote shell), tests/tools/fake_slurm plays sbatch / squeue / sacct /
// scancel and starts the real worker (app/python) on 127.0.0.1. No real ssh
// is run and no host but this one is contacted.
//
// Needs a Python ($SIRIUS_PYTHON, else host::findPython) and bash
// ($SIRIUS_TEST_BASH, else Git's bash on Windows, bash on PATH elsewhere);
// the end-to-end case also numpy in that Python. Cases skip without them.

#include <algorithm>
#include <atomic>
#include <chrono>
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

#include "core/cluster.hpp"
#include "core/errors.hpp"
#include "core/host.hpp"
#include "core/process.hpp"
#include "core/remote_host.hpp"
#include "core/remote_source.hpp"
#include "core/rpc.hpp"
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
            // the tools as LF scripts, whatever the checkout's line endings
            for (const char* tool : {"sbatch", "squeue", "sacct", "scancel"}) {
                std::string text = readAll(fs::path(SIRIUS_TEST_FAKE_SLURM_DIR) / tool);
                std::string lf;
                for (char c : text)
                    if (c != '\r') lf.push_back(c);
                std::ofstream(bin / tool, std::ios::binary) << lf;
            }
            std::ofstream(bin / "python3", std::ios::binary) << "#!/bin/bash\nexec \"$FAKE_PYTHON\" \"$@\"\n";
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
