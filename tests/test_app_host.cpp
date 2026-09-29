// Tests of the operating system without a window (app/core/host.hpp): the
// user's directories by the rules of each platform, temporary directories,
// whole files written atomically, executables on PATH, moving and removing
// trees (and refusing to remove the working directory), processes and the
// executable's own directory. Variables, the umask and the working directory
// are changed in-process only, and put back when each case ends.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>
#endif

#include "core/host.hpp"

using namespace sirius::app;
namespace fs = std::filesystem;

namespace {

    // Sets a variable (or, with nullopt, removes it) for as long as it
    // lives, then puts back what was there.
    class ScopedVariable {
    public:
        ScopedVariable(const char* name, std::optional<std::string> value) : name_(name) {
            if (host::hasEnvironment(name)) saved_ = host::environment(name);
            set(value);
        }
        ~ScopedVariable() { set(saved_); }
        ScopedVariable(const ScopedVariable&) = delete;
        ScopedVariable& operator=(const ScopedVariable&) = delete;

    private:
        void set(const std::optional<std::string>& value) const {
#ifdef _WIN32
            // Changes the process's environment as well as the C runtime's
            // copy; "" removes the variable.
            (void)_putenv_s(name_.c_str(), value ? value->c_str() : "");
#else
            if (value) ::setenv(name_.c_str(), value->c_str(), 1);
            else ::unsetenv(name_.c_str());
#endif
        }

        std::string name_;
        std::optional<std::string> saved_;
    };

    // A directory of the case's own, removed with what it holds.
    struct TempDir {
        std::string path = host::makeTempDirectory("sirius-host-test-");
        TempDir() { REQUIRE_FALSE(path.empty()); }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(fs::u8path(path), ec);
        }
        TempDir(const TempDir&) = delete;
        TempDir& operator=(const TempDir&) = delete;
    };

#ifdef _WIN32
    constexpr char kPathSeparator = ';';
    const std::string kExe = ".exe";
#else
    constexpr char kPathSeparator = ':';
    const std::string kExe;
#endif

} // namespace

TEST_CASE("host: the data and config directories follow each platform's rules", "[app][host]") {
#ifdef _WIN32
    {
        // A user name outside the ANSI code page comes back as UTF-8.
        const ScopedVariable local("LOCALAPPDATA", std::string("C:\\Users\\J\xC3\xBCrgen\\AppData\\Local"));
        CHECK(host::dataDirectory() == "C:/Users/J\xC3\xBCrgen/AppData/Local");
    }
    {
        const ScopedVariable local("LOCALAPPDATA", std::nullopt);
        const ScopedVariable profile("USERPROFILE", std::string("C:\\Users\\someone"));
        CHECK(host::dataDirectory() == "C:/Users/someone/AppData/Local");
    }
    {
        const ScopedVariable roaming("APPDATA", std::string("C:\\Users\\someone\\AppData\\Roaming"));
        CHECK(host::configDirectory() == "C:/Users/someone/AppData/Roaming");
    }
#elif defined(__APPLE__)
    const ScopedVariable home("HOME", std::string("/Users/someone"));
    CHECK(host::dataDirectory() == "/Users/someone/Library/Application Support");
#else
    const ScopedVariable home("HOME", std::string("/home/someone"));
    {
        const ScopedVariable xdg("XDG_DATA_HOME", std::string("/data/xdg"));
        CHECK(host::dataDirectory() == "/data/xdg");
    }
    {
        // Relative is invalid by the XDG specification, and ignored.
        const ScopedVariable xdg("XDG_DATA_HOME", std::string("relative/xdg"));
        CHECK(host::dataDirectory() == "/home/someone/.local/share");
    }
    {
        const ScopedVariable xdg("XDG_DATA_HOME", std::nullopt);
        CHECK(host::dataDirectory() == "/home/someone/.local/share");
        const ScopedVariable config("XDG_CONFIG_HOME", std::nullopt);
        CHECK(host::configDirectory() == "/home/someone/.config");
    }
#endif
}

TEST_CASE("host: environment tells a set variable from an unset one", "[app][host]") {
    const char* name = "SIRIUS_TEST_HOST_VARIABLE";
    {
        const ScopedVariable value(name, std::string("two words J\xC3\xBCrgen"));
        CHECK(host::hasEnvironment(name));
        CHECK(host::environment(name) == "two words J\xC3\xBCrgen");
    }
    CHECK_FALSE(host::hasEnvironment(name));
    CHECK(host::environment(name).empty());
#ifndef _WIN32
    // Windows' C runtime cannot set a variable to "" (that removes it).
    const ScopedVariable empty(name, std::string());
    CHECK(host::hasEnvironment(name));
    CHECK(host::environment(name).empty());
#endif
}

TEST_CASE("host: makeTempDirectory gives a new directory on every call", "[app][host]") {
    const std::string a = host::makeTempDirectory("sirius-host-test-");
    const std::string b = host::makeTempDirectory("sirius-host-test-");
    REQUIRE_FALSE(a.empty());
    REQUIRE_FALSE(b.empty());
    CHECK(a != b);
    CHECK(host::isDirectory(a));
    CHECK(host::isDirectory(b));
    CHECK(a.find('\\') == std::string::npos);
    CHECK(host::removeTree(a));
    CHECK(host::removeTree(b));
}

TEST_CASE("host: writeFileAtomic replaces a file whole and leaves nothing beside it", "[app][host]") {
    const TempDir dir;
    const std::string path = dir.path + "/settings.json";
    std::string text;
    CHECK_FALSE(host::readFile(path, text));
    REQUIRE(host::writeFileAtomic(path, "first"));
    REQUIRE(host::readFile(path, text));
    CHECK(text == "first");
    const std::string binary("second\0with a zero\r\n", 20);
    REQUIRE(host::writeFileAtomic(path, binary));
    REQUIRE(host::readFile(path, text));
    CHECK(text == binary);
    int entries = 0;
    for (const auto& entry : fs::directory_iterator(fs::u8path(dir.path))) {
        (void)entry;
        ++entries;
    }
    CHECK(entries == 1);   // no temporary file left behind
    CHECK(host::isFile(path));
    CHECK_FALSE(host::isDirectory(path));
    CHECK_FALSE(host::writeFileAtomic(dir.path + "/missing/settings.json", "x"));
    CHECK_FALSE(host::isDirectory(dir.path + "/missing"));
#ifndef _WIN32
    const std::string secret = dir.path + "/secret.json";
    REQUIRE(host::writeFileAtomic(secret, "token", true));
    struct stat st{};
    REQUIRE(::stat(secret.c_str(), &st) == 0);
    CHECK((st.st_mode & 0777) == 0600);
    // The mode is the file's from its creation, not what the umask allows.
    struct OpenUmask {
        const mode_t saved = ::umask(0);
        ~OpenUmask() { ::umask(saved); }
    } openUmask;
    REQUIRE(host::writeFileAtomic(secret, "another token", true));
    REQUIRE(::stat(secret.c_str(), &st) == 0);
    CHECK((st.st_mode & 0777) == 0600);
    REQUIRE(host::writeFileAtomic(path, "for everyone"));
    REQUIRE(::stat(path.c_str(), &st) == 0);
    CHECK((st.st_mode & 0777) == 0666);
#endif
}

TEST_CASE("host: findExecutable finds a program once its directory is on PATH", "[app][host]") {
    const TempDir dir;
    const std::string tool = "sirius-host-test-tool";
    const std::string file = dir.path + "/" + tool + kExe;
    REQUIRE(host::writeFileAtomic(file, "not really a program"));
#ifndef _WIN32
    REQUIRE(::chmod(file.c_str(), 0755) == 0);
#endif
    CHECK(host::findExecutable(tool).empty());
    {
        const ScopedVariable path("PATH", dir.path + kPathSeparator + host::environment("PATH"));
        CHECK(host::findExecutable(tool) == file);
        CHECK(host::findExecutable(tool + kExe) == file);
#ifndef _WIN32
        // Not executable: not a program.
        REQUIRE(::chmod(file.c_str(), 0644) == 0);
        CHECK(host::findExecutable(tool).empty());
        REQUIRE(::chmod(file.c_str(), 0755) == 0);
#endif
    }
    // A name with a directory is only checked.
    CHECK(host::findExecutable(file) == file);
    CHECK(host::findExecutable(dir.path + "/missing" + kExe).empty());
    CHECK(host::findExecutable("").empty());
#ifdef _WIN32
    // The Store's app execution aliases are never taken for programs.
    const std::string aliases = dir.path + "/Microsoft/WindowsApps";
    REQUIRE(host::makePath(aliases));
    REQUIRE(host::writeFileAtomic(aliases + "/sirius-host-test-alias.exe", ""));
    const ScopedVariable path("PATH", aliases + ";" + host::environment("PATH"));
    CHECK(host::findExecutable("sirius-host-test-alias").empty());
#endif
}

TEST_CASE("host: findPython gives an interpreter that exists, or nothing", "[app][host]") {
    const std::string python = host::findPython();
    if (python.empty()) SUCCEED("no Python on this machine");
    else {
        CHECK(host::isFile(python));
        CHECK(python.find('\\') == std::string::npos);
    }
}

TEST_CASE("host: renamePath and removeTree move and delete whole trees", "[app][host]") {
    const TempDir dir;
    const std::string tree = dir.path + "/tree";
    REQUIRE(host::makePath(tree + "/a/b"));
    REQUIRE(host::writeFileAtomic(tree + "/a/b/file.txt", "x"));
    REQUIRE(host::writeFileAtomic(tree + "/top.txt", "y"));
    // A read-only file (pip's RECORD files can be) does not stop the removal.
    fs::permissions(fs::u8path(tree + "/top.txt"), fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write,
                    fs::perm_options::remove);

    std::string error;
    CHECK(host::renamePath(tree, dir.path + "/moved", &error));
    INFO(error);
    CHECK_FALSE(host::isDirectory(tree));
    CHECK(host::isFile(dir.path + "/moved/a/b/file.txt"));
    error.clear();
    CHECK_FALSE(host::renamePath(dir.path + "/missing", dir.path + "/other", &error));
    CHECK_FALSE(error.empty());

    error.clear();
    CHECK(host::removeTree(dir.path + "/moved", &error));
    CHECK(error.empty());
    CHECK_FALSE(host::isDirectory(dir.path + "/moved"));
    CHECK(host::removeTree(dir.path + "/never-there"));
    // An empty path would otherwise mean the working directory.
    CHECK_FALSE(host::removeTree("", &error));
    CHECK_FALSE(error.empty());
    CHECK_FALSE(host::makePath(""));
}

TEST_CASE("host: removeTree refuses the working directory and what holds it, by any spelling", "[app][host]") {
    // Run from a directory of the case's own, two levels down, so that a
    // refusal that does not happen removes nothing but the case's own.
    const TempDir dir;
    const std::string inner = dir.path + "/a/b";
    REQUIRE(host::makePath(inner + "/c"));
    std::error_code ec;
    const fs::path previous = fs::current_path(ec);
    REQUIRE_FALSE(ec);
    fs::current_path(fs::u8path(inner), ec);
    REQUIRE_FALSE(ec);
    struct Back {
        const fs::path& to;
        ~Back() {
            std::error_code e;
            fs::current_path(to, e);
        }
    } back{previous};

    std::vector<std::string> refused{".", "./", "..", "../", "../..", "c/..", "c/../..", "./c/../", "c/./../"};
#ifdef _WIN32
    // "C:." is the working directory of drive C:, which is this one's.
    const std::string drive = fs::u8path(inner).root_name().u8string();
    refused.push_back(drive + ".");
    refused.push_back(drive + "c/..");
#endif
    for (const std::string& p : refused) {
        INFO(p);
        std::string error;
        CHECK_FALSE(host::removeTree(p, &error));
        CHECK(error.find("refusing") != std::string::npos);
    }
    CHECK(host::isDirectory(inner + "/c"));

    // A relative path that still names a directory below is removed as any.
    CHECK(host::removeTree("c/../c"));
    CHECK_FALSE(host::isDirectory(inner + "/c"));
    CHECK(host::isDirectory(inner));
}

#ifdef _WIN32
TEST_CASE("host: removeTree refuses a share or drive root spelt with more than one name", "[app][host]") {
    // Only servers and a drive that do not exist are named, so that a
    // refusal that does not happen fails to find anything to remove.
    std::string drive;
    for (char letter = 'Z'; letter >= 'D' && drive.empty(); --letter) {
        std::error_code ec;
        const std::string candidate = std::string(1, letter) + ":";
        if (fs::status(fs::path(candidate + "\\"), ec).type() == fs::file_type::not_found) drive = candidate;
    }
    std::vector<std::string> refused{"//sirius-no-such-server-4711/share",
                                     "\\\\sirius-no-such-server-4711\\share\\",
                                     "//sirius-no-such-server-4711/share/x/..",
                                     "\\\\?\\UNC\\sirius-no-such-server-4711\\share",
                                     "//?/UNC/sirius-no-such-server-4711/share/"};
    if (!drive.empty()) {
        refused.push_back("\\\\?\\" + drive + "\\");
        refused.push_back("\\\\.\\" + drive + "\\x\\..");
        refused.push_back("\\??\\" + drive + "\\");
    }
    for (const std::string& p : refused) {
        INFO(p);
        std::string error;
        CHECK_FALSE(host::removeTree(p, &error));
        CHECK(error.find("refusing") != std::string::npos);
    }

    // What is below such a root is removed as any.
    const TempDir dir;
    const std::string inner = dir.path + "/inner";
    REQUIRE(host::makePath(inner));
    std::string absolute = fs::absolute(fs::u8path(inner)).u8string();
    CHECK(host::removeTree("\\\\?\\" + absolute));
    CHECK_FALSE(host::isDirectory(inner));
}
#endif

TEST_CASE("host: processAlive knows this process and no invalid one", "[app][host]") {
    CHECK(host::processId() > 0);
    CHECK(host::processAlive(host::processId()));
    CHECK_FALSE(host::processAlive(0));
    CHECK_FALSE(host::processAlive(-1));
}

TEST_CASE("host: executableDirectory is the running test binary's directory", "[app][host]") {
    const std::string dir = host::executableDirectory();
    REQUIRE_FALSE(dir.empty());
    CHECK(host::isDirectory(dir));
    CHECK(fs::u8path(dir).is_absolute());
    CHECK(dir.find('\\') == std::string::npos);
    // The case runs in its own binary and in the one that links them all.
    CHECK((host::isFile(dir + "/test_app_host" + kExe) || host::isFile(dir + "/sirius_tests" + kExe)));
}
