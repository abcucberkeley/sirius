// "Open folder as dataset" on the cluster (core/cluster_folder.hpp): a folder
// of TIFF stacks named as an acquisition names them, parsed by a filename
// pattern into channels, time points and tiles, gives the same dataset
// whether the folder is on this computer or on the cluster -- listed through
// the fake cluster's SSH session (tests/tools/fake_ssh.py), the stacks'
// shapes asked of a real `sirius-cli serve` on 127.0.0.1, the manifest kept in
// ~/.sirius/manifests there and never in the (read-only) data folder -- and a
// pipeline file opens it again.
//
// No real ssh is run and no host but this one is contacted.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <sirius/tiff_io.hpp>

#include "core/array_source.hpp"
#include "core/cluster.hpp"
#include "core/cluster_folder.hpp"
#include "core/host.hpp"
#include "core/manifest.hpp"
#include "core/ops/load.hpp"
#include "core/process.hpp"
#include "core/remote_host.hpp"
#include "core/remote_source.hpp"
#include "core/rpc.hpp"
#include "core/workbench.hpp"
#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

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
        return {};
#else
        return host::findExecutable("bash");
#endif
    }

    void writeScript(const fs::path& path, const std::string& text) {
        std::ofstream(path, std::ios::binary) << text;
        std::error_code ec;
        fs::permissions(path, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::add, ec);
    }

    // A temporary "cluster home" behind the fake ssh, with the fake Slurm
    // tools (the login asks for the partitions) and a python3 on its PATH.
    struct FakeCluster {
        std::string python, bash;
        fs::path root, home, bin, slurm;

        bool usable() const { return !python.empty() && !bash.empty(); }

        FakeCluster() : python(testPython()), bash(testBash()) {
            root = sirius::test::uniqueTempPath("clfolder", "");
            home = root / "home";
            bin = root / "bin";
            slurm = root / "slurm";
            fs::create_directories(home);
            fs::create_directories(bin);
            fs::create_directories(slurm);
            for (const char* tool : {"sbatch", "srun", "squeue", "sacct", "scancel", "sinfo", "sacctmgr", "scontrol"}) {
                std::string lf;
                for (char c : readAll(fs::path(SIRIUS_TEST_FAKE_SLURM_DIR) / tool))
                    if (c != '\r') lf.push_back(c);
                writeScript(bin / tool, lf);
            }
            writeScript(bin / "python3", "#!/bin/bash\nexec \"$FAKE_PYTHON\" \"$@\"\n");
            setEnv("FAKE_SSH_LOG", (root / "ssh.log").generic_string());
            setEnv("FAKE_SSH_HOME", home.generic_string());
            setEnv("FAKE_SSH_PATH", bin.string());
            setEnv("FAKE_SSH_BASH", bash);
            setEnv("FAKE_SSH_USER", "tester");
            setEnv("FAKE_PYTHON", python);
            setEnv("FAKE_SLURM_DIR", slurm.generic_string());
            setEnv("FAKE_SSH_PROMPTS", "[]");
            setEnv("FAKE_SSH_ANSWERS", "[]");
            setEnv("USER", "tester");
        }
        ~FakeCluster() {
            std::error_code ec;
            for (const auto& e : fs::recursive_directory_iterator(root, ec))
                fs::permissions(e.path(), fs::perms::owner_write, fs::perm_options::add, ec);
            fs::remove_all(root, ec);
        }
    };

    template <typename F> bool waitFor(F done, std::chrono::seconds limit) {
        const auto end = std::chrono::steady_clock::now() + limit;
        while (std::chrono::steady_clock::now() < end) {
            if (done()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return done();
    }

    // A z-stack of uint16, values that say which file and voxel they are.
    void writeStack(const fs::path& path, Index z, Index y, Index x, int seed) {
        Buffer<std::uint16_t> stack(Shape{z, y, x});
        for (Index i = 0; i < z * y * x; ++i) stack.data()[i] = static_cast<std::uint16_t>((i * 7 + seed * 131) % 4093);
        writeTiffStack<std::uint16_t>(path.string(), stack.view(), TiffWriteOptions{});
    }

    // As the acquisition software names them: camera A, channel ch0 / ch1
    // (488 / 560 nm), stack (= time point) 0..2, two tiles in z, the time
    // stamp in the name.
    std::string acquisitionName(int ch, int stack, int z) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "10ms_CamA_ch%d_CAM1_stack%04d_%dnm_%07dmsec_%010dmsecAbs_000x_000y_%03dz_%04dt.tif", ch, stack,
                      ch == 0 ? 488 : 560, stack * 2500, 1700000000 + stack * 2500, z, stack);
        return buf;
    }

    constexpr const char* kPattern =
        R"(^10ms_CamA_ch(?P<channel>\d+)_CAM\d+_stack(?P<t>\d+)_\d+nm_\d+msec_\d+msecAbs_\d+x_\d+y_(?P<z>\d+)z_\d+t\.tiff?$)";

    // The acquisition: 2 channels x 3 time points x 2 tiles, 5 x 24 x 20 each,
    // written in an order that is not the natural one; and two files the
    // pattern leaves out.
    fs::path writeAcquisition(const fs::path& parent) {
        const fs::path acq = parent / "acq_2026";
        fs::create_directories(acq);
        int seed = 0;
        for (int stack = 2; stack >= 0; --stack)
            for (int ch = 0; ch < 2; ++ch)
                for (int z = 0; z < 2; ++z) writeStack(acq / acquisitionName(ch, stack, z), 5, 24, 20, ++seed);
        writeStack(acq / "overview_MIP.tif", 1, 24, 20, 99);
        std::ofstream(acq / "settings.txt") << "laser 488 560\n";
        return acq;
    }

    FilenameRule acquisitionRule() {
        FilenameRule r;
        r.pattern = kPattern;
        r.positions = FilenameRule::Positions::GridIndex;
        r.voxelUm = {0.104, 0.104, 0.3};
        r.frameIntervalS = 2.5;
        r.acquisition = "AOLLS";
        ChannelInfo gfp;
        gfp.label = "GFP";
        gfp.wavelengthNm = 488;
        r.channelInfo["0"] = gfp;
        ChannelInfo mch;
        mch.label = "mCherry";
        mch.wavelengthNm = 560;
        r.channelInfo["1"] = mch;
        return r;
    }

    std::set<std::string> entriesOf(const fs::path& dir) {
        std::set<std::string> out;
        for (const auto& e : fs::directory_iterator(dir)) out.insert(e.path().filename().string());
        return out;
    }

    // Read-only, as data on a cluster often is: the files and the folder.
    void makeReadOnly(const fs::path& dir) {
        for (const auto& e : fs::directory_iterator(dir))
            fs::permissions(e.path(), fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write, fs::perm_options::remove);
        fs::permissions(dir, fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write, fs::perm_options::remove);
    }

    // The manifest without where it is: what a pattern made of the names.
    json mapping(const DatasetManifest& m) {
        json j = m.toJson();
        j.erase("files_folder");
        return j;
    }

    // The dataset the cluster's manifest opens is this computer's: the same
    // dims, order, voxel size, channels, tiles and voxels; and a pipeline
    // file keeps the cluster:// manifest and opens the same dataset again.
    void checkSameDataset(const fs::path& root, const std::string& localManifest, const std::string& source) {
        Workbench local(root / "wb-local");
        local.openDataset(localManifest);
        REQUIRE(local.hasDataset());
        Workbench node(root / "wb-node");
        node.openDataset(source);
        REQUIRE(node.hasDataset());
        const DatasetMeta& a = local.dataset();
        const DatasetMeta& b = node.dataset();
        CHECK(a.dims.toString() == b.dims.toString());
        CHECK(b.dims.c == 2);
        CHECK(b.dims.t == 3);
        CHECK(b.dims.z == 5);
        for (std::size_t k = 0; k < 3; ++k) CHECK(std::abs(a.voxelUm[k] - b.voxelUm[k]) < 1e-9);
        REQUIRE(a.channels.size() == b.channels.size());
        for (std::size_t c = 0; c < a.channels.size(); ++c) CHECK(a.channels[c].label == b.channels[c].label);
        CHECK(b.channels[0].label == "GFP");
        CHECK(a.tiles.size() == b.tiles.size());
        OpenOptions lazy;
        lazy.readAll = false;
        const OpenResult ra = openDataset(localManifest, lazy), rb = openDataset(source, lazy);
        REQUIRE(ra.source);
        REQUIRE(rb.source);
        const Index plane = a.dims.y * a.dims.x;
        std::vector<float> pa(static_cast<std::size_t>(plane)), pb(static_cast<std::size_t>(plane));
        for (Index c = 0; c < 2; ++c)
            for (Index t = 0; t < 3; ++t) {
                ra.source->readPlane(c, t, 3, pa.data());
                rb.source->readPlane(c, t, 3, pb.data());
                CHECK(std::memcmp(pa.data(), pb.data(), pa.size() * sizeof(float)) == 0);
            }

    // the pipeline file keeps the cluster:// manifest and opens the same dataset again
        const std::string file = (root / "acq.sirius.toml").generic_string();
        node.savePipeline(file);
        CHECK(readAll(file).find(source) != std::string::npos);
        Workbench back(root / "wb-back");
        back.loadPipeline(file);
        CHECK(back.pipeline().at(0).params.getString("path") == source);
        REQUIRE(back.hasDataset());
        CHECK(back.dataset().dims.toString() == a.dims.toString());
        CHECK(back.dataset().channels.size() == a.channels.size());
    }

} // namespace

TEST_CASE("cluster folder: the listing, the names in natural order, the manifest's name and its text", "[app][cluster][manifest]") {
    // what the folder script answers, read
    const std::string answer =
        "noise before\n"
        R"({"path": "/data/acq", "home": "/home/u", "tiffs": ["f10.tif", "f2.tif", "f1.TIF", ".hidden.tif"], "others": 3, "truncated": true, )"
        R"("store": false, "manifest": "format = \"sirius-dataset\"\nversion = 1\nname = \"acq\"\npattern = \"^f(?P<t>\\\\d+)\"\n"})"
        "\n";
    const ClusterFolder f = parseClusterFolder(answer);
    CHECK(f.path == "/data/acq");
    CHECK(f.home == "/home/u");
    CHECK(f.tiffs == std::vector<std::string>{"f1.TIF", "f2.tif", "f10.tif"});
    CHECK(f.others == 3);
    CHECK(f.truncated);
    REQUIRE(f.existing);
    CHECK(f.existing->pattern == "^f(?P<t>\\d+)");
    CHECK_THROWS_AS(parseClusterFolder(R"({"error": "/x: Permission denied"})"), ssh::SshError);
    // the script lists by a shell word, and asks for nothing but names
    const std::string script = clusterFolderScript("~/acq 1", 100);
    CHECK(script.find("\"$HOME\"/'acq 1' 100") != std::string::npos);
    CHECK(script.find("os.scandir") != std::string::npos);

    // one name per host and folder, readable, the same every time
    const std::string a = clusterManifestName("fiona", "/data/acq_2026");
    CHECK(a == clusterManifestName("fiona", "/data/acq_2026"));
    CHECK(a.rfind("acq_2026-", 0) == 0);
    CHECK(a.size() == std::string("acq_2026-").size() + 16 + 5);
    CHECK(a != clusterManifestName("fiona", "/other/acq_2026"));
    CHECK(a != clusterManifestName("other", "/data/acq_2026"));
    CHECK(clusterManifestName("h", "/data/a b;$(x)").find_first_of(" ;$()") == std::string::npos);
    // the manifest's text goes as base64: no quote of it reaches a shell
    const std::string w = writeClusterManifestScript("name = \"it's\"\n", "acq-1.toml");
    CHECK(w.find("it's") == std::string::npos);
    CHECK(w.find("'acq-1.toml'") != std::string::npos);

    // the TOML text and back
    const fs::path dir = sirius::test::uniqueTempPath("clfolder_local", "");
    fs::create_directories(dir);
    const fs::path acq = writeAcquisition(dir);
    const DatasetManifest m = manifestFromFolder(acq, acquisitionRule());
    CHECK(mapping(DatasetManifest::fromText(m.toText(), "x.toml")) == mapping(m));
    // over the same names, with the same shapes, the same manifest as the folder's
    const DatasetManifest byNames = manifestFromNames(
        "acq_2026", tiffNamesOf({acquisitionName(1, 0, 0), "overview_MIP.tif", acquisitionName(0, 0, 0), acquisitionName(0, 0, 1), acquisitionName(1, 0, 1), acquisitionName(0, 1, 0), acquisitionName(0, 1, 1), acquisitionName(1, 1, 0), acquisitionName(1, 1, 1), acquisitionName(0, 2, 0), acquisitionName(0, 2, 1), acquisitionName(1, 2, 0), acquisitionName(1, 2, 1), "settings.txt"}),
        acquisitionRule(), [](const std::string&) { return StackShape{20, 24, 5}; });
    CHECK(mapping(byNames) == mapping(m));
    REQUIRE(m.channels.size() == 2);
    CHECK(m.channels[0].label == "GFP");
    CHECK(m.channels[1].label == "mCherry");
    CHECK(m.timePoints() == 3);
    CHECK(m.tiles.size() == 2);
    CHECK(m.files.size() == 12);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("cluster folder: a folder parsed by a pattern on the cluster opens as on this computer; the data folder is not written; a "
          "pipeline file opens it again",
          "[app][cluster][engine][manifest]") {
    FakeCluster fc;
    if (!fc.usable()) SKIP("no Python or bash for the fake ssh");
    const fs::path acq = writeAcquisition(fc.home / "data");
    const std::set<std::string> before = entriesOf(acq);
    makeReadOnly(acq);
    const FilenameRule rule = acquisitionRule();

    // this computer: the dialog's manifest, kept outside the folder (files_folder)
    std::vector<std::string> unmatchedHere;
    DatasetManifest here = manifestFromFolder(acq, rule, &unmatchedHere);
    CHECK(unmatchedHere == std::vector<std::string>{"overview_MIP.tif"});
    const fs::path localManifest = fc.root / "local-manifests" / "acq_2026.toml";
    fs::create_directories(localManifest.parent_path());
    here.filesFolder = acq.generic_string();
    here.save(localManifest);

    // the cluster: the login only (no job), the names over its command channel
    cluster::Session session;
    session.setAskpassProgram(SIRIUS_TEST_ASKPASS);
    cluster::Profile p;
    p.host = "fakecluster";
    p.sshProgram = fc.python;
    p.sshProgramArgs = {SIRIUS_TEST_FAKE_SSH};
    session.logIn(p);
    REQUIRE(waitFor([&] { return session.status().state != cluster::State::Connecting; }, std::chrono::seconds(60)));
    INFO(session.status().reason);
    REQUIRE(session.sshUp());
    const ClusterFolder folder = listClusterFolder(session, "fakecluster", "~/data/acq_2026");
    CHECK(folder.tiffs == tiffNamesInOrder(acq));
    CHECK(folder.others == 1);
    CHECK_FALSE(folder.truncated);
    CHECK_FALSE(folder.existing);
    CHECK(fs::equivalent(fs::path(folder.path), acq));
    // the preview: the same matches, the same groups
    const std::vector<FilenameMatch> there = matchFilenames(folder.tiffs, rule.pattern), mine = matchFilenames(tiffNamesInOrder(acq), rule.pattern);
    REQUIRE(there.size() == mine.size());
    for (std::size_t i = 0; i < there.size(); ++i) {
        CHECK(there[i].file == mine[i].file);
        CHECK(there[i].matched == mine[i].matched);
        CHECK(there[i].groups == mine[i].groups);
    }

    // the engine on the "node": a real sirius-cli serve on 127.0.0.1
    const std::string tokenFile = (fc.root / "token").string();
    std::ofstream(tokenFile) << "folder-token-3\n";
    fs::permissions(tokenFile, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
    ChildProcess serve;
    ChildProcess::Options o;
    o.program = SIRIUS_TEST_CLI;
    o.arguments = {"serve", "--port", "0", "--no-python-worker", "--device", "cpu", "--exit-with-parent", "--scratch", (fc.root / "scratch").string()};
    o.environment = {{"SIRIUS_TOKEN_FILE", tokenFile}};
    o.killTree = true;
    REQUIRE(serve.start(o));
    std::string line;
    REQUIRE(serve.readLine(line, 60000));
    RemoteConfig rc;
    rc.host = "127.0.0.1";
    rc.port = json::parse(line)["port"].get<int>();
    rc.token = "folder-token-3";
    auto datasets = std::make_shared<RemoteDatasets>("fakecluster", [rc] { return rc.open(); });
    datasets->install();

    // the manifest from the cluster's names and the node's shapes: the same mapping
    std::vector<std::string> unmatchedThere;
    const DatasetManifest built = manifestFromClusterFolder(folder, rule, &unmatchedThere);
    CHECK(unmatchedThere == unmatchedHere);
    CHECK(mapping(built) == mapping(here));
    const std::string source = writeClusterManifest(session, folder, built);
    std::string host, remotePath;
    REQUIRE(splitClusterPath(source, host, remotePath));
    CHECK(host == "fakecluster");
    INFO(remotePath);
    CHECK(fs::equivalent(fs::path(remotePath).parent_path(), fc.home / ".sirius" / "manifests"));
    CHECK(fs::path(remotePath).filename().string() == clusterManifestName("fakecluster", folder.path));
    const DatasetManifest kept = DatasetManifest::load(fs::path(remotePath));
    CHECK(fs::equivalent(fs::path(kept.filesFolder), acq));
    CHECK(kept.pattern == kPattern);
    // nothing was added to (or changed in) the read-only data folder
    CHECK(entriesOf(acq) == before);
    CHECK(loadSourceIsFolder(source));

    checkSameDataset(fc.root, localManifest.generic_string(), source);
    CHECK(entriesOf(acq) == before);

    datasets->uninstall();
    session.disconnect(false);
}
