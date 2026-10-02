// Tests of child processes with their streams on pipes (app/core/process.hpp),
// run against the helper tests/tools/sirius_test_child: lines both ways, exit
// codes, stopping, a program that is not there, the PATH and environment the
// child gets, stderr merged into readLine, ending a whole process tree, and on
// Linux starting children while other threads keep the C library's locks
// busy. Every case leaves no process behind.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "core/host.hpp"
#include "core/process.hpp"

#ifndef SIRIUS_TEST_CHILD
#error "SIRIUS_TEST_CHILD names the helper executable (tests/app_tests.cmake)"
#endif

using namespace sirius::app;
using Clock = std::chrono::steady_clock;

namespace {

    const std::string kChild = SIRIUS_TEST_CHILD;

    ChildProcess::Options child(std::vector<std::string> arguments) {
        ChildProcess::Options o;
        o.program = kChild;
        o.arguments = std::move(arguments);
        return o;
    }

    // Every line readLine gives until it reports the end (or ten seconds
    // pass without one).
    std::vector<std::string> readAll(ChildProcess& p) {
        std::vector<std::string> lines;
        std::string line;
        while (p.readLine(line, 10000)) lines.push_back(line);
        return lines;
    }

    // The first stdout line of a helper run with these options.
    std::string firstLine(const ChildProcess::Options& options) {
        ChildProcess p;
        std::string error;
        REQUIRE(p.start(options, &error));
        std::string line;
        CHECK(p.readLine(line, 10000));
        CHECK(p.waitForExit(10000));
        return line;
    }

    bool endsWithin(int pid, int ms) {
        const auto deadline = Clock::now() + std::chrono::milliseconds(ms);
        while (host::processAlive(pid)) {
            if (Clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return true;
    }

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
            (void)_putenv_s(name_.c_str(), value ? value->c_str() : "");
#else
            if (value) ::setenv(name_.c_str(), value->c_str(), 1);
            else ::unsetenv(name_.c_str());
#endif
        }

        std::string name_;
        std::optional<std::string> saved_;
    };

} // namespace

TEST_CASE("process: readLine gives the child's stdout lines in order, then the end", "[app][process]") {
    ChildProcess p;
    std::string error;
    REQUIRE(p.start(child({"print", "one", "two words", "three"}), &error));
    CHECK(readAll(p) == std::vector<std::string>{"one", "two words", "three"});
    CHECK(p.waitForExit(10000));
    CHECK(p.exitCode() == 0);
    std::string line;
    CHECK_FALSE(p.readLine(line, 0));
}

TEST_CASE("process: waitForExit gives the exit code, and times out while the child runs", "[app][process]") {
    ChildProcess p;
    REQUIRE(p.start(child({"exit", "7"})));
    CHECK(p.waitForExit(10000));
    CHECK(p.exitCode() == 7);
    CHECK_FALSE(p.running());

    REQUIRE(p.start(child({"sleep", "30000"})));
    CHECK(p.exitCode() == -1);
    CHECK_FALSE(p.waitForExit(50));
    CHECK(p.running());
    p.stop(0);
    CHECK_FALSE(p.running());
}

TEST_CASE("process: writeInput reaches the child's stdin, and closeInput ends it", "[app][process]") {
    ChildProcess p;
    REQUIRE(p.start(child({"echo-stdin"})));
    CHECK(p.writeInput("first\nsecond line\n"));
    std::string line;
    REQUIRE(p.readLine(line, 10000));
    CHECK(line == "first");
    REQUIRE(p.readLine(line, 10000));
    CHECK(line == "second line");
    CHECK(p.writeInput("third\n"));
    REQUIRE(p.readLine(line, 10000));
    CHECK(line == "third");
    p.closeInput();
    CHECK(p.waitForExit(10000));
    CHECK(p.exitCode() == 0);
    CHECK_FALSE(p.writeInput("late\n"));

    // A child that has ended reads nothing more: the write fails, and on
    // POSIX raises no SIGPIPE that would end this test binary.
    REQUIRE(p.start(child({"exit", "0"})));
    REQUIRE(p.waitForExit(10000));
    CHECK_FALSE(p.writeInput(std::string(1 << 20, 'x')));
    CHECK_FALSE(p.writeInput("again\n"));
}

TEST_CASE("process: stop ends a child that does not leave by itself", "[app][process]") {
    ChildProcess p;
    REQUIRE(p.start(child({"sleep", "60000"})));
    const auto started = Clock::now();
    p.stop(400);
    CHECK(Clock::now() - started < std::chrono::seconds(5));
    CHECK_FALSE(p.running());
    CHECK(p.exitCode() != 0);
    p.stop();   // again, with nothing running
}

TEST_CASE("process: a program that does not exist is reported as not found", "[app][process]") {
    ChildProcess p;
    std::string error;
    ChildProcess::Options missing;
    missing.program = "sirius-no-such-program-4711";
    CHECK_FALSE(p.start(missing, &error));
    CHECK(p.programNotFound());
    CHECK_FALSE(error.empty());

    const std::string dir = std::filesystem::u8path(kChild).parent_path().u8string();
    missing.program = dir + "/sirius-no-such-program-4711";
    error.clear();
    CHECK_FALSE(p.start(missing, &error));
    CHECK(p.programNotFound());
    CHECK_FALSE(error.empty());

    REQUIRE(p.start(child({"exit", "0"})));
    CHECK_FALSE(p.programNotFound());
    CHECK(p.waitForExit(10000));

    // A working directory that does not exist is another failure.
    ChildProcess::Options elsewhere = child({"exit", "0"});
    elsewhere.workingDirectory = dir + "/sirius-no-such-directory-4711";
    error.clear();
    CHECK_FALSE(p.start(elsewhere, &error));
    CHECK_FALSE(p.programNotFound());
    CHECK_FALSE(error.empty());
}

TEST_CASE("process: a program without a directory is looked up on the PATH the child gets", "[app][process]") {
    const std::string dir = std::filesystem::u8path(kChild).parent_path().u8string();
    ChildProcess::Options o;
    o.program = "sirius_test_child";
    o.arguments = {"exit", "5"};
    o.environment = {{"PATH", dir}};
    ChildProcess p;
    std::string error;
    REQUIRE(p.start(o, &error));
    CHECK(p.waitForExit(10000));
    CHECK(p.exitCode() == 5);

    o.environment.clear();
    o.unsetEnvironment = {"PATH"};
    CHECK_FALSE(p.start(o, &error));
    CHECK(p.programNotFound());
}

TEST_CASE("process: a relative program path is taken from the child's working directory", "[app][process]") {
    const std::filesystem::path helper = std::filesystem::u8path(kChild);
    ChildProcess::Options o = child({"exit", "6"});
    o.program = "./" + helper.filename().u8string();
    o.workingDirectory = helper.parent_path().u8string();
    ChildProcess p;
    std::string error;
    REQUIRE(p.start(o, &error));
    CHECK(p.waitForExit(10000));
    CHECK(p.exitCode() == 6);
}

TEST_CASE("process: the child's environment can lose, empty and replace variables", "[app][process]") {
    const char* name = "SIRIUS_TEST_PROCESS_VARIABLE";
    const ScopedVariable inherited(name, std::string("inherited"));
    CHECK(firstLine(child({"env", name})) == "inherited");

    ChildProcess::Options unset = child({"env", name});
    unset.unsetEnvironment = {name};
    CHECK(firstLine(unset) == "<unset>");

    ChildProcess::Options empty = child({"env", name});
    empty.environment = {{name, ""}};
    CHECK(firstLine(empty).empty());

    ChildProcess::Options replaced = child({"env", name});
    replaced.environment = {{name, "replaced"}};
    CHECK(firstLine(replaced) == "replaced");

    // What the child is given wins over what it would not inherit.
    replaced.unsetEnvironment = {name};
    CHECK(firstLine(replaced) == "replaced");

#ifdef _WIN32
    // Names are case-insensitive on Windows.
    ChildProcess::Options lower = child({"env", name});
    lower.unsetEnvironment = {"sirius_test_process_variable"};
    CHECK(firstLine(lower) == "<unset>");
#endif
}

TEST_CASE("process: mergeErrorLines queues stderr lines for readLine as well", "[app][process]") {
    std::mutex m;
    std::vector<std::string> handled;
    ChildProcess p;
    p.setErrorHandler([&](const std::string& line) {
        const std::lock_guard<std::mutex> g(m);
        handled.push_back(line);
    });

    ChildProcess::Options merged = child({"print", "out", "--stderr", "err one", "err two"});
    merged.mergeErrorLines = true;
    REQUIRE(p.start(merged));
    const std::vector<std::string> lines = readAll(p);   // ends once both streams have
    CHECK(p.waitForExit(10000));
    p.stop(0);   // joins the readers: every handler call has happened
    REQUIRE(lines.size() == 3);
    CHECK(std::count(lines.begin(), lines.end(), "out") == 1);
    const auto one = std::find(lines.begin(), lines.end(), "err one");
    const auto two = std::find(lines.begin(), lines.end(), "err two");
    REQUIRE(one != lines.end());
    REQUIRE(two != lines.end());
    CHECK(one < two);   // one stream keeps its order
    {
        const std::lock_guard<std::mutex> g(m);
        CHECK(handled == std::vector<std::string>{"err one", "err two"});
        handled.clear();
    }

    // Without it, stderr reaches the handler only.
    REQUIRE(p.start(child({"print", "out", "--stderr", "err one", "err two"})));
    CHECK(readAll(p) == std::vector<std::string>{"out"});
    CHECK(p.waitForExit(10000));
    p.stop(0);
    const std::lock_guard<std::mutex> g(m);
    CHECK(handled == std::vector<std::string>{"err one", "err two"});
}

TEST_CASE("process: killTree ends what the child started as well", "[app][process]") {
    std::string line;
    // On Windows stop() returns once the grandchild has gone, its files
    // closed: a cancelled setup removes the folder it ran from right after.
    // Elsewhere the group is signalled, and the grandchild's new parent
    // collects it a moment later.
    const auto gone = [](int pid) {
#ifdef _WIN32
        return !host::processAlive(pid);
#else
        return endsWithin(pid, 5000);
#endif
    };
    SECTION("stopped while the child runs") {
        ChildProcess p;
        ChildProcess::Options o = child({"spawn-grandchild", "30000"});
        o.killTree = true;
        std::vector<std::string> notes;
        p.setErrorHandler([&notes](const std::string& note) { notes.push_back(note); });   // read once stop() has joined the readers
        REQUIRE(p.start(o));
        REQUIRE(p.readLine(line, 10000));
        const int grandchild = std::stoi(line);
        CHECK(host::processAlive(grandchild));
        p.stop(200);
        // Where no job can hold the child (this runs inside a job that
        // allows no nesting), the note says so and only the child ends, as
        // documented; the grandchild then leaves by itself later.
        if (!notes.empty()) SKIP("no job could hold the child: " + notes.front());
        CHECK(gone(grandchild));
    }
    SECTION("stopped after the child has left by itself") {
        ChildProcess p;
        ChildProcess::Options o = child({"spawn-grandchild", "30000", "0"});
        o.killTree = true;
        REQUIRE(p.start(o));
        REQUIRE(p.readLine(line, 10000));
        const int grandchild = std::stoi(line);
        CHECK(p.waitForExit(10000));
        CHECK(p.exitCode() == 0);
        CHECK(host::processAlive(grandchild));
        p.stop(0);
        CHECK(gone(grandchild));
    }
    SECTION("without it, only the child ends") {
        // The grandchild outlives the child and then leaves by itself, so
        // that this case, too, leaves nothing running.
        ChildProcess p;
        REQUIRE(p.start(child({"spawn-grandchild", "3000"})));
        REQUIRE(p.readLine(line, 10000));
        const int grandchild = std::stoi(line);
        p.stop(0);
        CHECK_FALSE(p.running());
        CHECK(host::processAlive(grandchild));
        CHECK(endsWithin(grandchild, 15000));
    }
}

TEST_CASE("process: killTree without the parent-death signal outlives the starting thread", "[app][process]") {
    // ssh is started this way: on a connect thread that ends long before the
    // session does. The child must live on until stop(), which still ends it.
    ChildProcess p;
    ChildProcess::Options o = child({"sleep", "30000"});
    o.killTree = true;
    o.parentDeathSignal = false;
    bool started = false;
    std::thread([&] { started = p.start(o); }).join();
    REQUIRE(started);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CHECK(p.running());
    p.stop(0);
    CHECK_FALSE(p.running());
}

TEST_CASE("process: ownProcessGroup starts the child in a group of its own", "[app][process]") {
    ChildProcess::Options own = child({"ids"});
    own.ownProcessGroup = true;
    const std::string ownIds = firstLine(own);
#ifdef _WIN32
    CHECK_FALSE(ownIds.empty());
#else
    const std::string plainIds = firstLine(child({"ids"}));
    const auto group = [](const std::string& ids) { return std::stol(ids.substr(ids.find(' ') + 1)); };
    const auto pid = [](const std::string& ids) { return std::stol(ids.substr(0, ids.find(' '))); };
    CHECK(group(ownIds) == pid(ownIds));
    CHECK(group(plainIds) == static_cast<long>(::getpgrp()));
#endif
}

#ifdef __linux__
// The child used to call setenv between fork and exec. With another thread
// inside setenv or malloc at the moment of the fork, the child waited
// forever for a lock nobody would release in it; now it only execs what was
// prepared before the fork.
TEST_CASE("process: children start while eight other threads keep the C library's locks busy", "[app][process]") {
    const ScopedVariable busy("SIRIUS_TEST_PROCESS_BUSY", std::string("0"));
    std::atomic<bool> done{false};
    std::vector<std::thread> threads;
    struct Join {
        std::atomic<bool>& done;
        std::vector<std::thread>& threads;
        ~Join() {
            done.store(true);
            for (std::thread& t : threads) t.join();
        }
    } join{done, threads};
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&done, t] {
            unsigned i = 0;
            while (!done.load(std::memory_order_relaxed)) {
                // An existing variable, from a few values: glibc keeps every
                // value it was given, and replaces the entry in place.
                const std::string value = std::to_string((static_cast<unsigned>(t) * 7 + i++) % 16);
                ::setenv("SIRIUS_TEST_PROCESS_BUSY", value.c_str(), 1);
                std::vector<char> scratch(1024 + i % 4096);
                scratch.front() = static_cast<char>(i);
            }
        });
    }
    for (int n = 0; n < 50; ++n) {
        ChildProcess p;
        ChildProcess::Options o = child({"exit", "3"});
        o.environment = {{"SIRIUS_TEST_PROCESS_RUN", std::to_string(n)}};
        o.unsetEnvironment = {"SIRIUS_TEST_PROCESS_BUSY"};
        std::string error;
        REQUIRE(p.start(o, &error));
        REQUIRE(p.waitForExit(10000));
        CHECK(p.exitCode() == 3);
    }
}
#endif
