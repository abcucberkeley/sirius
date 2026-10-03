// The settings file (core/settings_toml, core/settings_store): TOML both
// ways for every kind of value the settings hold, the JSON file of before
// migrated (and kept as <name>.migrated, its secrets moved out), two
// instances writing the same file, a file that does not read (the defaults,
// the file left alone), the settings editor's save, and its checks of the
// cluster profiles (core/cluster_profiles).

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "core/cluster_profiles.hpp"
#include "core/settings_store.hpp"
#include "core/settings_toml.hpp"
#include "temp_path.hpp"

using namespace sirius::app;
using json = nlohmann::json;

namespace {

    namespace fs = std::filesystem;

    std::string readAll(const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        std::stringstream s;
        s << in.rdbuf();
        return s.str();
    }

    void writeAll(const fs::path& p, const std::string& text) {
        std::ofstream out(p, std::ios::binary);
        out << text;
    }

    // A settings folder of its own, removed afterwards.
    struct Dir {
        fs::path path = sirius::test::uniqueTempPath("settings", "");
        Dir() { fs::create_directories(path); }
        ~Dir() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
        fs::path file(const char* name) const { return path / name; }
    };

    // What every settings test starts from: the store on `dir`.
    Settings& store(const Dir& d) {
        Settings& s = Settings::instance();
        s.setDirectory(d.path.generic_u8string());
        return s;
    }

    std::size_t count(const std::string& text, const std::string& what) {
        std::size_t n = 0;
        for (std::size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
        return n;
    }

} // namespace

TEST_CASE("settings: every kind of value goes to TOML and comes back as it was", "[app][settings]") {
    const json flat = {
        {"window/width", 1500},
        {"window/maximized", false},
        {"ui/scale", 1.25},
        {"ui/whole", 2.0},   // a float that looks whole stays a float
        {"compute/big", 9007199254740993LL},
        {"compute/negative", -42},
        {"compute/tiny", 1e-300},
        {"compute/tenth", 0.1},
        {"worker/python", "C:\\Users\\Zoë\\AppData\\Local\\sirius\\python-env\\Scripts\\python.exe"},
        {"worker/quote", "say \"hi\" 'there'"},
        {"worker/lines", "one\ntwo\n\tthree"},
        {"worker/empty", ""},
        {"recent/datasets", {"C:/data/a b.tif", "/home/u/µm/stack.tif"}},
        {"recent/none", json::array()},
        {"recent/numbers", {1, 2, 3}},
        {"folderDataset/patternMap", {{"C:/data/x y", "(?<z>\\d+)"}, {"/home/u/[weird].dir", "p"}}},
        {"cluster/current", "lab"},
        {"cluster/lab",
         {{"host", "login.example.org"},
          {"binds", {"/data", "/scratch/u:/scratch:ro"}},
          {"job", {{"partition", "gpu"}, {"gpus", 1}}},
          {"partitions", {{{"name", "gpu"}, {"default", true}, {"qos", {"normal"}}}, {{"name", "cpu"}, {"accounts", {"lab", "other"}}}}}}},
        {"group/deeper/key", "a slash past the first is part of the key"},
        {"toplevel", 7},
    };
    const std::string text = settings_toml::toToml(flat);
    INFO(text);
    const settings_toml::ParseResult back = settings_toml::fromToml(text);
    REQUIRE(back.ok);
    CHECK(back.flat == flat);
    CHECK(back.flat["ui/whole"].is_number_float());
    CHECK(back.flat["window/width"].is_number_integer());
    // the sections a person reads, with SIRIUS's notes above its own
    CHECK(text.rfind("# SIRIUS settings.", 0) == 0);
    CHECK(text.find("Comments added by\n# hand are not kept") != std::string::npos);
    CHECK(text.find("[cluster.lab]") != std::string::npos);
    CHECK(text.find("[[cluster.lab.partitions]]") != std::string::npos);
    CHECK(text.find("# Clusters: one [cluster.<name>] table per cluster profile") != std::string::npos);
    // written twice, the same text: the order is stable
    CHECK(settings_toml::toToml(back.flat) == text);
}

TEST_CASE("settings: secrets, nulls and bytes that are not UTF-8 never reach the file", "[app][settings]") {
    const json flat = {{"secrets/hpc/token", "QUJD"}, {"worker/python", nullptr}, {"recent/bad", std::string("a\xFF"
                                                                                                             "b\xC3")},
                       {"worker/dir", "x"}};
    const std::string text = settings_toml::toToml(flat);
    CHECK(text.find("secrets") == std::string::npos);
    CHECK(text.find("QUJD") == std::string::npos);
    const settings_toml::ParseResult back = settings_toml::fromToml(text);
    REQUIRE(back.ok);
    CHECK_FALSE(back.flat.contains("worker/python"));
    CHECK(back.flat["recent/bad"] == "a\xEF\xBF\xBD"
                                     "b\xEF\xBF\xBD");
    CHECK(back.flat["worker/dir"] == "x");
    CHECK(settings_toml::validUtf8("ok \xC3\xA9 \xE2\x82\xAC") == "ok \xC3\xA9 \xE2\x82\xAC");
    CHECK(settings_toml::validUtf8("\xC0\x80") == "\xEF\xBF\xBD\xEF\xBF\xBD");   // an overlong NUL
}

TEST_CASE("settings: a file that is not TOML says where", "[app][settings]") {
    const settings_toml::ParseResult r = settings_toml::fromToml("[worker]\npython = \"a\"\n[window\nwidth = 3\n");
    CHECK_FALSE(r.ok);
    CHECK(r.line == 3);
    CHECK(r.column > 0);
    CHECK_FALSE(r.error.empty());
    CHECK(r.flat.empty());
    // and where a value is, for the editor
    const std::string ok = "[worker]\npython = \"a\"\n\n[cluster.lab]\nhost = \"x\"\n\n[[cluster.lab.partitions]]\nname = \"gpu\"\n";
    const auto at = settings_toml::position(ok, {"cluster", "lab", "partitions", "0", "name"});
    REQUIRE(at);
    CHECK(at->first == 8);
    CHECK(settings_toml::position(ok, {"cluster", "lab", "nothing"})->first <= 5);   // the nearest part that is there
    CHECK(settings_toml::splitPath({"cluster", "lab", "job", "gpus"}) ==
          std::pair<std::string, std::vector<std::string>>{"cluster/lab", {"job", "gpus"}});
}

TEST_CASE("settings: the JSON file of before becomes the TOML file, kept as .migrated, its secrets moved out", "[app][settings]") {
    for (const char* oldName : {"sirius-app.json", "sirius-imgui.json"}) {
        Dir d;
        const json old = {{"worker/python", "C:/py/python.exe"},
                          {"window/width", 1200},
                          {"secrets/hpc/token", "RFBBUEk="},
                          {"cluster/profile", {{"host", "login.example.org"}, {"partition", "gpu"}, {"container", "~/w.sif"}}}};
        writeAll(d.file(oldName), old.dump(2));
        Settings& s = store(d);
        CHECK(s.getString("worker/python") == "C:/py/python.exe");
        CHECK(s.getInt("window/width") == 1200);
        REQUIRE(s.save());
        CHECK(fs::exists(d.file("sirius-app.toml")));
        CHECK_FALSE(fs::exists(d.file(oldName)));
        CHECK(fs::exists(d.path / (std::string(oldName) + ".migrated")));
        const std::string toml = readAll(d.file("sirius-app.toml"));
        INFO(toml);
        CHECK(toml.find("secrets") == std::string::npos);
        CHECK(toml.find("RFBBUEk=") == std::string::npos);
        CHECK(toml.find("C:/py/python.exe") != std::string::npos);
        // the secret, where the secret store reads it
        const json secrets = json::parse(readAll(d.file("secrets.json")));
        CHECK(secrets["hpc/token"] == "RFBBUEk=");
        // a second start reads the TOML file, not the old one
        Settings& again = store(d);
        CHECK(again.getString("worker/python") == "C:/py/python.exe");
        CHECK(again.loadError().empty());
    }
}

TEST_CASE("settings: two instances writing the file keep each other's settings", "[app][settings]") {
    Dir d;
    Settings& s = store(d);
    s.set("worker/python", "mine");
    s.set("window/width", 900);
    REQUIRE(s.save());
    // the other instance: it adds a key of its own and changes one of ours
    settings_toml::ParseResult r = settings_toml::fromToml(readAll(d.file("sirius-app.toml")));
    REQUIRE(r.ok);
    r.flat["assistant/model"] = "theirs";
    r.flat["window/width"] = 1000;
    writeAll(d.file("sirius-app.toml"), settings_toml::toToml(r.flat));
    // this one changes another key, without having looked
    s.set("compute/backend", "CPU");
    REQUIRE(s.save());
    const settings_toml::ParseResult file = settings_toml::fromToml(readAll(d.file("sirius-app.toml")));
    REQUIRE(file.ok);
    CHECK(file.flat["assistant/model"] == "theirs");   // theirs kept
    CHECK(file.flat["window/width"] == 1000);          // their change kept: this one did not change it again
    CHECK(file.flat["worker/python"] == "mine");
    CHECK(file.flat["compute/backend"] == "CPU");
    // and this instance reads theirs
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    CHECK(s.getString("assistant/model") == "theirs");
    // a removal here reaches the file without taking theirs
    s.remove("worker/python");
    REQUIRE(s.save());
    const settings_toml::ParseResult after = settings_toml::fromToml(readAll(d.file("sirius-app.toml")));
    CHECK_FALSE(after.flat.contains("worker/python"));
    CHECK(after.flat["assistant/model"] == "theirs");
}

TEST_CASE("settings: a file that does not read gives the defaults and is left as it is", "[app][settings]") {
    Dir d;
    const std::string broken = "[worker]\npython = \"C:/py\"\n[window\nwidth = 3\n";
    writeAll(d.file("sirius-app.toml"), broken);
    Settings& s = store(d);
    CHECK(s.getString("worker/python", "default") == "default");
    const std::string why = s.loadError();
    CHECK(why.find("line 3") != std::string::npos);
    // changes stay in memory; the file is not written over
    s.set("compute/backend", "CPU");
    CHECK(s.getString("compute/backend") == "CPU");
    CHECK_FALSE(s.save());
    CHECK(readAll(d.file("sirius-app.toml")) == broken);
    // the editor's save of text that does not read: refused, nothing written
    std::string error;
    CHECK_FALSE(s.adoptText("[a\n", &error));
    CHECK(error.find("line 1") != std::string::npos);
    CHECK(readAll(d.file("sirius-app.toml")) == broken);
    // of text that reads: written as it is (comments too), taken at once
    const std::string fixed = "# mine\n[worker]\npython = \"C:/py\"  # kept as typed\n";
    REQUIRE(s.adoptText(fixed, &error));
    CHECK(readAll(d.file("sirius-app.toml")) == fixed);
    CHECK(s.getString("worker/python") == "C:/py");
    CHECK(s.loadError().empty());
    // what was set before the fix is dropped: the text is what the user wants
    CHECK(s.getString("compute/backend", "none") == "none");
    // and the next save writes again
    s.set("window/width", 800);
    CHECK(s.save());
    CHECK(settings_toml::fromToml(readAll(d.file("sirius-app.toml"))).flat["window/width"] == 800);
}

TEST_CASE("settings: the editor's checks of the cluster profiles say what is wrong and where", "[app][settings][cluster]") {
    using cluster::checkSettingsText;
    const std::string good = "[cluster]\ncurrent = \"lab\"\n\n[cluster.lab]\nhost = \"login.example.org\"\nimage = \"/images/w.sif\"\n"
                             "binds = [\"/data\"]\n\n[cluster.lab.job]\npartition = \"gpu\"\ntime = \"01:00:00\"\ngpus = 1\nmem = \"64G\"\n\n"
                             "[[cluster.lab.partitions]]\nname = \"gpu\"\ndefault = true\nqos = [\"normal\"]\nmax_time = \"3-00:00:00\"\n";
    CHECK(checkSettingsText(good).empty());

    // not TOML: one error, where toml++ stopped
    auto p = checkSettingsText("[cluster.lab\nhost = \"x\"\n");
    REQUIRE(p.size() == 1);
    CHECK(p[0].error);
    CHECK(p[0].line == 1);
    CHECK(p[0].message.find("not valid TOML") != std::string::npos);

    auto has = [](const std::vector<cluster::SettingsProblem>& ps, const std::string& what, bool error, int line) {
        for (const cluster::SettingsProblem& x : ps)
            if (x.message.find(what) != std::string::npos && x.error == error && (line == 0 || x.line == line)) return true;
        return false;
    };
    // no host, a number as text, an unknown key, a partition without a name, a time Slurm does not read
    const std::string bad = "[cluster.lab]\nimage = \"\"\nfavourite = 3\n\n[cluster.lab.job]\ngpus = \"two\"\ntime = \"1:xx\"\nmem = \"lots\"\n\n"
                            "[[cluster.lab.partitions]]\nqos = [\"normal\"]\n";
    p = checkSettingsText(bad);
    INFO(p.size());
    CHECK(has(p, "needs host", true, 0));
    CHECK(has(p, "No worker image yet", false, 2));                     // empty image: said, not refused
    CHECK(has(p, "has no key \"favourite\"", true, 3));
    CHECK(has(p, "gpus must be a whole number", true, 6));
    CHECK(has(p, "is not a time Slurm reads", true, 7));
    CHECK(has(p, "is not a size Slurm reads", true, 8));
    CHECK(has(p, "This partition has no name", true, 0));
    // an older key is read all the same, and said
    p = checkSettingsText("[cluster.old]\nhost = \"h\"\ncontainer = \"/w.sif\"\nvenv = \"~/v\"\n");
    CHECK(has(p, "\"container\" is an older name", false, 3));
    CHECK(has(p, "nothing: the worker runs in the image", false, 4));
    for (const auto& x : p) CHECK_FALSE(x.error);
    // a current that names no profile
    CHECK(has(checkSettingsText("[cluster]\ncurrent = \"gone\"\n"), "names no profile", false, 2));
}

TEST_CASE("settings: the example of cluster profiles in docs/ reads and passes the checks", "[app][settings][cluster]") {
    const std::string text = readAll(fs::path(SIRIUS_TEST_SOURCE_DIR) / "docs" / "clusters.example.toml");
    REQUIRE_FALSE(text.empty());
    const std::vector<cluster::SettingsProblem> problems = cluster::checkSettingsText(text);
    for (const cluster::SettingsProblem& p : problems) FAIL_CHECK(p.line << ": " << p.message);
    CHECK(problems.empty());
    const cluster::ProfileBook book = cluster::ProfileBook::fromSettings(settings_toml::fromToml(text).flat);
    CHECK(book.profiles.size() == 2);
    CHECK(book.current == "mycluster");
    REQUIRE(book.find("mycluster"));
    CHECK(book.find("mycluster")->choices.size() == 2);
}
