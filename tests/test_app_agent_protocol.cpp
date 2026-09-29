// The JSON-lines session and the MCP server over a tool dispatcher
// (app/core/agent_protocol.hpp), driven with a fake dispatcher: lines go in
// through receive() as the CLI's stdin reader would feed them, and every line
// the server writes is collected and parsed back. The "slow" tool blocks until
// its request is cancelled, which is how the reader-thread paths (ping,
// status, cancel) are shown to work while the main thread is busy.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/agent_protocol.hpp"

using namespace sirius::app::agent;
using json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

    using Clock = std::chrono::steady_clock;

    constexpr const char* kWorkspace = "ws_0123456789ab";
    const std::vector<std::uint8_t> kPngSignature = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};

    // Waits for a condition another thread makes true.
    bool waitUntil(const std::function<bool()>& condition, std::chrono::milliseconds timeout = 5s) {
        const auto deadline = Clock::now() + timeout;
        while (!condition()) {
            if (Clock::now() > deadline) return false;
            std::this_thread::sleep_for(1ms);
        }
        return true;
    }

    // Every line the server writes, from whichever thread wrote it.
    class Output {
    public:
        LineSink sink() {
            return [this](const std::string& line) {
                std::lock_guard<std::mutex> lock(mutex_);
                lines_.push_back(line);
                changed_.notify_all();
            };
        }
        std::vector<std::string> lines() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return lines_;
        }
        std::size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return lines_.size();
        }
        // Parsed; a line that does not parse comes back as {"unparsed": line}.
        std::vector<json> messages() const {
            std::vector<json> out;
            for (const std::string& line : lines()) {
                json m = json::parse(line, nullptr, false);
                out.push_back(m.is_discarded() ? json{{"unparsed", line}} : m);
            }
            return out;
        }
        bool waitFor(const std::function<bool(const json&)>& match, std::chrono::milliseconds timeout = 5s) const {
            std::unique_lock<std::mutex> lock(mutex_);
            return changed_.wait_for(lock, timeout, [&] {
                return std::any_of(lines_.begin(), lines_.end(), [&](const std::string& line) {
                    const json m = json::parse(line, nullptr, false);
                    return !m.is_discarded() && match(m);
                });
            });
        }

    private:
        mutable std::mutex mutex_;
        mutable std::condition_variable changed_;
        std::vector<std::string> lines_;
    };

    // Numbers and strings are different ids even when they read alike. (The
    // kind of number is not compared: a parsed 1 is unsigned, json(1) signed.)
    bool sameId(const json& a, const json& b) {
        return a == b && ((a.is_number() && b.is_number()) || (a.is_string() && b.is_string()));
    }
    // A response (session or MCP) to the request with this id.
    std::function<bool(const json&)> answers(json id) {
        return [id](const json& m) {
            return m.is_object() && m.contains("id") && sameId(m["id"], id) && !m.contains("event") && !m.contains("method");
        };
    }
    // What the server wrote is read through non-const json: the const
    // operator[] asserts on a key that is not there, which in a Debug build
    // would end the whole test binary instead of failing one check, while
    // the non-const one adds a null that the check then compares.
    json* find(std::vector<json>& messages, const std::function<bool(const json&)>& match) {
        const auto it = std::find_if(messages.begin(), messages.end(), match);
        return it == messages.end() ? nullptr : &*it;
    }
    // The response to that id; the test case fails here, rather than crashing
    // on what it reads next, when there is none. The id is taken by value, so
    // that a reference to the response never looks bound to a temporary.
    json& answerTo(std::vector<json>& messages, json id) {
        json* found = find(messages, answers(id));
        if (!found) FAIL("no response to the request with id " << id.dump());
        return *found;
    }
    std::ptrdiff_t indexOf(const std::vector<json>& messages, const std::function<bool(const json&)>& match) {
        const auto it = std::find_if(messages.begin(), messages.end(), match);
        return it == messages.end() ? -1 : it - messages.begin();
    }
    std::vector<json> events(const std::vector<json>& messages, const std::string& kind) {
        std::vector<json> out;
        for (const json& m : messages)
            if (m.is_object() && m.value("event", std::string()) == kind) out.push_back(m);
        return out;
    }

    // What the tests call and inspect. call(), pump() and takeEvents() run on
    // the server's main thread; status() and cancelActive() on any.
    class FakeDispatcher final : public ToolDispatcher {
    public:
        std::atomic<bool> slowStarted{false}, slowRunning{false}, slowEnded{false};
        std::atomic<int> cancels{0};
        std::atomic<bool> runActive{false}, runCancelled{false};
        json lastArgs;
        // Set holdNextTools and the next tools() blocks, with toolsHeld set,
        // until releaseTools is (at most 5 s): a tools/list the client can
        // cancel while it is being answered.
        mutable std::atomic<bool> holdNextTools{false}, toolsHeld{false}, releaseTools{false};

        std::vector<ToolDescriptor> tools() const override {
            if (holdNextTools.exchange(false)) {
                toolsHeld = true;
                waitUntil([this] { return releaseTools.load(); });
            }
            const auto tool = [](std::string name, std::string title, bool readOnly, json meta = json::object()) {
                ToolDescriptor t;
                t.name = std::move(name);
                t.title = std::move(title);
                t.description = "The " + t.name + " tool of the fake dispatcher.";
                t.inputSchema = {{"type", "object"}, {"properties", json::object()}, {"additionalProperties", false}};
                t.hints.readOnly = readOnly;
                t.hints.idempotent = readOnly;
                t.meta = std::move(meta);
                return t;
            };
            return {tool("echo", "Echo", true, {{"anthropic/maxResultSizeChars", 200000}}),
                    tool("fail", "Fail", true),
                    tool("image", "Image", true),
                    tool("slow", "Slow", false),
                    tool("progress", "Progress", true),
                    tool("log", "Log", true),
                    tool("start_run", "Start a run", false),
                    tool("run_status", "Run status", true),
                    tool("throws", "Throws", false)};
        }
        bool hasTool(const std::string& name) const override {
            for (const ToolDescriptor& t : tools())
                if (t.name == name) return true;
            return false;
        }
        ToolResult call(const std::string& name, const json& args, const CallContext& ctx) override {
            lastArgs = args;
            if (name == "echo") {
                ToolResult r;
                r.value = args;
                r.changes = {"echoed"};
                r.undoable = true;
                if (args.contains("warn")) r.warnings.push_back("argument 'warn' is not one of echo's and was ignored");
                return r;
            }
            if (name == "fail") return failure("unknown_step", "no step 99", "steps are numbered from 1", {{"step", 99}});
            if (name == "image") {
                ToolResult r;
                r.value = {{"width", 3}, {"height", 2}, {"path", "C:/scratch/renders/render-0001.png"}};
                Attachment a;
                a.mimeType = "image/png";
                a.bytes = kPngSignature;
                a.path = "C:/scratch/renders/render-0001.png";
                a.width = 3;
                a.height = 2;
                // "count" copies, for the limits on the images together.
                for (int i = 0, n = args.value("count", 1); i < n; ++i) r.images.push_back(a);
                return r;
            }
            if (name == "slow") {
                slowRunning = true;
                slowStarted = true;
                double f = 0;
                while (!ctx.cancelled()) {
                    f = std::min(1.0, f + 0.001);
                    ctx.progress(f, "Step 02 \xC2\xB7 Slow \xC2\xB7 waiting");
                    std::this_thread::sleep_for(1ms);
                }
                slowRunning = false;
                slowEnded = true;
                return failure("cancelled", "the slow tool was cancelled");
            }
            if (name == "progress") {
                for (const json& f : args.value("fractions", json::array()))
                    ctx.progress(f.get<double>(), "Step 02 \xC2\xB7 Contrast \xC2\xB7 working");
                return {};
            }
            if (name == "log") {
                std::lock_guard<std::mutex> lock(eventsMutex_);
                const std::string line = args.value("line", std::string("hello"));
                events_.push_back({{"event", "log"}, {"source", "workbench"}, {"line", line}});
                return {};
            }
            if (name == "start_run") {
                const int ms = args.value("ms", 100);
                runForever_ = ms < 0;
                runEndsAt_ = Clock::now() + std::chrono::milliseconds(std::max(ms, 0));
                runCancelled = false;
                runActive = true;
                ToolResult r;
                r.value = {{"status", "running"}, {"run_id", "r1"}};
                return r;
            }
            if (name == "run_status") {
                ToolResult r;
                r.value = {{"status", runActive ? "running" : "idle"}, {"run_id", "r1"}};
                return r;
            }
            if (name == "throws") throw std::runtime_error("the dispatcher broke");
            return failure("unknown_tool", "unknown tool '" + name + "'");
        }
        Status status() const override {
            Status s;
            s.running = runActive;
            s.workspace = kWorkspace;
            if (s.running) s.runId = "r1";
            return s;
        }
        void cancelActive() override {
            ++cancels;
            runCancelled = true;
        }
        void pump() override {
            if (runActive && (runCancelled || (!runForever_ && Clock::now() >= runEndsAt_))) {
                std::lock_guard<std::mutex> lock(eventsMutex_);
                const char* status = runCancelled ? "cancelled" : "succeeded";
                events_.push_back({{"event", "run_finished"}, {"run_id", "r1"}, {"result", {{"status", status}}}});
                runActive = false;
            }
        }
        std::vector<json> takeEvents() override {
            std::lock_guard<std::mutex> lock(eventsMutex_);
            std::vector<json> out;
            out.swap(events_);
            return out;
        }

    private:
        std::mutex eventsMutex_;
        std::vector<json> events_;
        Clock::time_point runEndsAt_;
        bool runForever_ = false;
    };

    ServerOptions serverOptions() {
        ServerOptions o;
        o.version = "9.9.9";
        o.progressInterval = 0ms;
        return o;
    }

    McpOptions mcpOptions() {
        McpOptions o;
        o.instructions = "Open a dataset, then run.";
        return o;
    }

    // Steps a server on this thread until it finishes; false when it does not
    // within the timeout.
    bool runToEnd(Server& server, std::chrono::milliseconds timeout = 10s) {
        const auto deadline = Clock::now() + timeout;
        while (server.step(5ms))
            if (Clock::now() > deadline) return false;
        return true;
    }

    // The server's main thread, for the cases where the test itself plays the
    // reader thread.
    class MainLoop {
    public:
        explicit MainLoop(Server& server) : server_(server) {
            thread_ = std::thread([this] {
                while (server_.step(5ms)) {}
                done_ = true;
            });
        }
        ~MainLoop() { join(); }
        // Ends the loop: after endOfInput(), or by terminating a server that
        // did not finish in time.
        bool join(std::chrono::milliseconds timeout = 10s) {
            if (!thread_.joinable()) return done_;
            const bool finished = waitUntil([this] { return done_.load(); }, timeout);
            if (!finished) server_.terminate();
            thread_.join();
            return finished;
        }

    private:
        Server& server_;
        std::atomic<bool> done_{false};
        std::thread thread_;
    };

    std::string mcpRequest(const json& id, const std::string& method, json params = json::object()) {
        return json{{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", std::move(params)}}.dump();
    }
    std::string initializeRequest(const std::string& version, const json& id = 0) {
        const json client = {{"name", "test"}, {"version", "1"}};
        const json params = {{"protocolVersion", version}, {"capabilities", json::object()}, {"clientInfo", client}};
        return mcpRequest(id, "initialize", params);
    }
    json modernMeta(const std::string& version = "2026-07-28") {
        return {{"io.modelcontextprotocol/protocolVersion", version},
                {"io.modelcontextprotocol/clientCapabilities", json::object()}};
    }

} // namespace

// --- shared ----------------------------------------------------------------------------

TEST_CASE("agent protocol: error codes map to the CLI's exit codes", "[app][agent_protocol]") {
    CHECK(exitCodeFor("") == 0);
    for (const char* code : {"failed", "run_failed", "export_failed", "io_error", "internal", "something_new"})
        CHECK(exitCodeFor(code) == 1);
    for (const char* code : {"usage", "unknown_tool"}) CHECK(exitCodeFor(code) == 2);
    for (const char* code : {"not_found", "open_failed", "invalid_argument", "unknown_step", "unknown_operation", "no_dataset",
                             "validation", "not_computed", "unsupported", "too_large", "busy", "stale_workspace"})
        CHECK(exitCodeFor(code) == 3);
    CHECK(exitCodeFor("worker_unavailable") == 4);
    CHECK(exitCodeFor("python_not_found") == 4);
    CHECK(exitCodeFor("consent_required") == 5);
    CHECK(exitCodeFor("timeout") == 124);
    CHECK(exitCodeFor("cancelled") == 130);

    ToolResult f = failure("busy", "a run is active", "wait for it", {{"run_id", "r2"}});
    CHECK_FALSE(f.ok);
    CHECK(f.error.code == "busy");
    CHECK(f.error.message == "a run is active");
    CHECK(f.error.hint == "wait for it");
    CHECK(f.error.data["run_id"] == "r2");
    CHECK(f.value.is_object());

    Status idle;
    idle.workspace = kWorkspace;
    json j = idle.toJson();
    CHECK(j["running"] == false);
    CHECK(j["step"].is_null());
    CHECK(j["run_id"].is_null());
    CHECK(j["active_tool"].is_null());
    CHECK(j["workspace"] == kWorkspace);
    Status busy;
    busy.running = true;
    busy.fraction = 0.5;
    busy.step = 2;
    busy.runId = "r3";
    busy.activeTool = "run";
    busy.message = "Step 02";
    json b = busy.toJson();
    CHECK(b["step"] == 2);
    CHECK(b["fraction"] == 0.5);
    CHECK(b["run_id"] == "r3");
    CHECK(b["active_tool"] == "run");
    CHECK(b["message"] == "Step 02");
}

// --- session ---------------------------------------------------------------------------

TEST_CASE("agent protocol: the session starts with a ready event", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());
    s.start();
    std::vector<json> m = out.messages();
    REQUIRE(m.size() == 1);
    CHECK(m[0]["event"] == "ready");
    CHECK(m[0]["protocol"] == "sirius-session/1");
    CHECK(m[0]["version"] == "9.9.9");
    CHECK(m[0]["workspace"] == kWorkspace);
    CHECK(m[0]["tools"] == d.tools().size());
}

TEST_CASE("agent protocol: session requests are answered in order with the tool's result", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());
    s.receive(R"({"id":1,"method":"echo","params":{"a":2,"warn":true}})");
    s.receive(R"({"id":"two","method":"fail"})");
    s.receive(R"({"id":3,"method":"tools","jsonrpc":"2.0"})");
    s.receive(R"({"id":4,"method":"image","params":{"inline":true}})");
    s.receive(R"({"id":5,"method":"image"})");
    s.receive(R"({"id":6,"method":"throws"})");
    s.endOfInput();
    REQUIRE(runToEnd(s));
    CHECK(s.exitCode() == 0);

    std::vector<json> m = out.messages();
    for (const std::string& line : out.lines()) CHECK(line.find('\n') == std::string::npos);
    REQUIRE(m.size() == 6);
    const json ids = json::array({1, "two", 3, 4, 5, 6});
    for (std::size_t i = 0; i < m.size(); ++i) CHECK(answers(ids[i])(m[i]));

    CHECK(m[0]["ok"] == true);
    CHECK(m[0]["result"] == json({{"a", 2}, {"warn", true}}));
    CHECK(m[0]["changes"] == json::array({"echoed"}));
    CHECK(m[0]["undoable"] == true);
    CHECK(m[0]["warnings"].size() == 1);

    CHECK(m[1]["ok"] == false);
    CHECK(m[1]["error"]["code"] == "unknown_step");
    CHECK(m[1]["error"]["message"] == "no step 99");
    CHECK(m[1]["error"]["hint"] == "steps are numbered from 1");
    CHECK(m[1]["error"]["data"]["step"] == 99);

    REQUIRE(m[2]["result"]["tools"].size() == d.tools().size());
    json& echo = m[2]["result"]["tools"][0];
    CHECK(echo["name"] == "echo");
    CHECK(echo["title"] == "Echo");
    CHECK(echo["inputSchema"]["type"] == "object");
    CHECK(echo["annotations"]["readOnlyHint"] == true);
    CHECK(echo["_meta"]["anthropic/maxResultSizeChars"] == 200000);

    // "inline" is the session's option, not the tool's.
    json& image = m[3]["result"]["image"];
    CHECK(image["path"] == "C:/scratch/renders/render-0001.png");
    CHECK(image["mime_type"] == "image/png");
    CHECK(image["width"] == 3);
    CHECK(image["height"] == 2);
    CHECK(image["bytes"] == 8);
    CHECK(image["base64"] == "iVBORw0KGgo=");
    CHECK(m[4]["result"]["image"]["path"] == "C:/scratch/renders/render-0001.png");
    CHECK_FALSE(m[4]["result"]["image"].contains("base64"));
    CHECK_FALSE(d.lastArgs.contains("inline"));

    CHECK(m[5]["ok"] == false);
    CHECK(m[5]["error"]["code"] == "internal");
}

TEST_CASE("agent protocol: malformed session requests get protocol errors", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());

    // Answered on the reader thread, before any step().
    s.receive("this is not json");
    s.receive("[1, 2]");
    s.receive(R"({"method":"echo"})");
    s.receive(R"({"id":null,"method":"echo"})");
    s.receive(R"({"id":7})");
    s.receive(R"({"id":8,"method":"echo","params":[1]})");
    s.receive("");
    s.receive("   \r");
    s.receive("{\"id\":10,\"method\":\"ping\"}\r");
    std::vector<json> m = out.messages();
    REQUIRE(m.size() == 7);
    CHECK(m[0]["id"].is_null());
    CHECK(m[0]["error"]["code"] == "parse_error");
    for (int i = 1; i <= 3; ++i) {
        CHECK(m[i]["id"].is_null());
        CHECK(m[i]["error"]["code"] == "invalid_request");
    }
    CHECK(m[4]["id"] == 7);
    CHECK(m[4]["error"]["code"] == "invalid_request");
    CHECK(m[5]["id"] == 8);
    CHECK(m[5]["error"]["code"] == "invalid_request");
    CHECK(m[6]["id"] == 10);
    CHECK(m[6]["ok"] == true);

    s.receive(R"({"id":9,"method":"no_such_thing"})");
    s.endOfInput();
    REQUIRE(runToEnd(s));
    m = out.messages();
    json* unknown = find(m, answers(9));
    REQUIRE(unknown);
    CHECK((*unknown)["ok"] == false);
    CHECK((*unknown)["error"]["code"] == "unknown_method");
    CHECK(exitCodeFor((*unknown)["error"]["code"].get<std::string>()) == 2);
}

TEST_CASE("agent protocol: session ping and status are answered while a tool runs, and cancel ends it", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());
    s.start();
    MainLoop loop(s);

    s.receive(R"({"id":1,"method":"slow"})");
    REQUIRE(waitUntil([&] { return d.slowStarted.load(); }));
    s.receive(R"({"id":2,"method":"ping"})");
    REQUIRE(out.waitFor(answers(2)));
    CHECK(d.slowRunning);   // the ping did not wait for the tool

    s.receive(R"({"id":3,"method":"status"})");
    REQUIRE(out.waitFor(answers(3)));
    std::vector<json> m = out.messages();
    json& status = answerTo(m, 3);
    CHECK(status["result"]["active_tool"] == "slow");
    CHECK(status["result"]["workspace"] == kWorkspace);
    CHECK(status["result"]["running"] == false);

    // A request queued behind the slow one is dropped by id, and answered as cancelled.
    s.receive(R"({"id":4,"method":"echo"})");
    s.receive(R"({"id":5,"method":"cancel","params":{"id":4}})");
    REQUIRE(out.waitFor(answers(5)));
    CHECK(d.slowRunning);
    m = out.messages();
    REQUIRE(find(m, answers(4)));
    CHECK(answerTo(m, 4)["error"]["code"] == "cancelled");
    CHECK(answerTo(m, 5)["result"]["cancelled"] == true);

    s.receive(R"({"id":6,"method":"cancel"})");
    REQUIRE(out.waitFor(answers(1)));
    REQUIRE(out.waitFor(answers(6)));
    CHECK(d.slowEnded);
    m = out.messages();
    CHECK(answerTo(m, 1)["error"]["code"] == "cancelled");
    CHECK(answerTo(m, 6)["result"]["cancelled"] == true);

    s.endOfInput();
    CHECK(loop.join());
    CHECK(s.exitCode() == 0);
}

TEST_CASE("agent protocol: session progress events are throttled and can be turned off", "[app][agent_protocol]") {
    const std::string request = R"({"id":1,"method":"progress","params":{"fractions":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}})";
    FakeDispatcher d;

    SECTION("an interval longer than the call lets one through") {
        Output out;
        ServerOptions o = serverOptions();
        o.progressInterval = std::chrono::hours(1);
        SessionServer s(d, out.sink(), o);
        s.receive(request);
        s.endOfInput();
        REQUIRE(runToEnd(s));
        std::vector<json> m = out.messages();
        std::vector<json> progress = events(m, "progress");
        REQUIRE(progress.size() == 1);
        CHECK(progress[0]["id"] == 1);
        CHECK(progress[0]["fraction"] == 0.1);
        CHECK(progress[0]["step"] == 2);
        CHECK(progress[0]["step_name"] == "Contrast");
        CHECK(progress[0]["message"] == "working");
        CHECK(indexOf(m, [](const json& e) { return e.value("event", "") == "progress"; }) < indexOf(m, answers(1)));
    }
    SECTION("no interval lets every one through") {
        Output out;
        SessionServer s(d, out.sink(), serverOptions());
        s.receive(request);
        s.endOfInput();
        REQUIRE(runToEnd(s));
        CHECK(events(out.messages(), "progress").size() == 9);
    }
    SECTION("subscribe {progress:false} turns them off") {
        Output out;
        SessionServer s(d, out.sink(), serverOptions());
        s.receive(R"({"id":0,"method":"subscribe","params":{"progress":false}})");
        s.receive(request);
        s.endOfInput();
        REQUIRE(runToEnd(s));
        std::vector<json> m = out.messages();
        CHECK(events(m, "progress").empty());
        CHECK(answerTo(m, 0)["result"] == json({{"progress", false}, {"log", false}}));
    }
}

TEST_CASE("agent protocol: session log events are written only after subscribe {log:true}", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());
    s.receive(R"({"id":1,"method":"log","params":{"line":"first"}})");
    s.receive(R"({"id":2,"method":"subscribe","params":{"log":true}})");
    s.receive(R"({"id":3,"method":"log","params":{"line":"second"}})");
    s.receive(R"({"id":4,"method":"subscribe","params":{"log":"yes"}})");
    s.endOfInput();
    REQUIRE(runToEnd(s));
    std::vector<json> m = out.messages();
    std::vector<json> logs = events(m, "log");
    REQUIRE(logs.size() == 1);
    CHECK(logs[0]["line"] == "second");
    CHECK(logs[0]["source"] == "workbench");
    // The call's log lines come before its answer.
    CHECK(indexOf(m, [](const json& e) { return e.value("event", "") == "log"; }) < indexOf(m, answers(3)));
    CHECK(answerTo(m, 2)["result"]["log"] == true);
    CHECK(answerTo(m, 4)["error"]["code"] == "invalid_argument");
}

TEST_CASE("agent protocol: session end of input waits for a running run and writes run_finished", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());
    s.receive(R"({"id":1,"method":"start_run","params":{"ms":150}})");
    s.endOfInput();
    REQUIRE(runToEnd(s));
    CHECK(s.exitCode() == 0);
    CHECK_FALSE(d.runActive);
    CHECK(d.cancels == 0);
    std::vector<json> m = out.messages();
    std::vector<json> finished = events(m, "run_finished");
    REQUIRE(finished.size() == 1);
    CHECK(finished[0]["result"]["status"] == "succeeded");
    CHECK(indexOf(m, answers(1)) < indexOf(m, [](const json& e) { return e.value("event", "") == "run_finished"; }));
}

TEST_CASE("agent protocol: session end of input cancels a run that outlives eofRunWait", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    ServerOptions o = serverOptions();
    o.eofRunWait = 50ms;
    SessionServer s(d, out.sink(), o);
    s.receive(R"({"id":1,"method":"start_run","params":{"ms":-1}})");
    s.endOfInput();
    const auto start = Clock::now();
    REQUIRE(runToEnd(s));
    CHECK(Clock::now() - start >= 50ms);
    CHECK(d.cancels >= 1);
    std::vector<json> finished = events(out.messages(), "run_finished");
    REQUIRE(finished.size() == 1);
    CHECK(finished[0]["result"]["status"] == "cancelled");
}

TEST_CASE("agent protocol: session end of input drains the queue, then step() returns false", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());
    for (int i = 1; i <= 5; ++i) s.receive(json{{"id", i}, {"method", "echo"}, {"params", {{"n", i}}}}.dump());
    s.endOfInput();
    s.receive(R"({"id":6,"method":"echo"})");   // after end of input: not read
    REQUIRE(runToEnd(s));
    CHECK_FALSE(s.step(0ms));
    std::vector<json> m = out.messages();
    REQUIRE(m.size() == 5);
    for (int i = 0; i < 5; ++i) {
        CHECK(m[i]["id"] == i + 1);
        CHECK(m[i]["result"]["n"] == i + 1);
    }
}

TEST_CASE("agent protocol: session shutdown answers what is queued behind it as cancelled", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());
    s.receive(R"({"id":1,"method":"echo"})");
    s.receive(R"({"id":2,"method":"shutdown"})");
    s.receive(R"({"id":3,"method":"echo"})");
    REQUIRE(runToEnd(s));
    CHECK(s.exitCode() == 0);
    std::vector<json> m = out.messages();
    REQUIRE(m.size() == 3);
    CHECK(m[0]["ok"] == true);
    CHECK(m[1]["id"] == 2);
    CHECK(m[1]["ok"] == true);
    CHECK(m[2]["id"] == 3);
    CHECK(m[2]["error"]["code"] == "cancelled");
}

TEST_CASE("agent protocol: after shutdown the session still answers status, ping and cancel", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());   // eofRunWait unset: the run is waited for without a bound
    MainLoop loop(s);
    s.receive(R"({"id":1,"method":"start_run","params":{"ms":-1}})");
    s.receive(R"({"id":2,"method":"shutdown"})");
    REQUIRE(out.waitFor(answers(2)));

    // The session waits for the run now; lines are still read. The reader
    // thread answers these at once, and anything else is cancelled.
    s.receive(R"({"id":3,"method":"status"})");
    s.receive(R"({"id":4,"method":"ping"})");
    s.receive(R"({"id":5,"method":"echo"})");
    s.receive(R"({"id":6,"method":"tools"})");
    std::vector<json> m = out.messages();
    CHECK(answerTo(m, 3)["result"]["running"] == true);
    CHECK(answerTo(m, 3)["result"]["run_id"] == "r1");
    CHECK(answerTo(m, 4)["ok"] == true);
    CHECK(answerTo(m, 5)["error"]["code"] == "cancelled");
    CHECK(answerTo(m, 6)["error"]["code"] == "cancelled");
    CHECK(d.cancels == 0);

    // Cancelling the run ends the wait, and the session with it.
    s.receive(R"({"id":7,"method":"cancel","params":{"id":1}})");
    CHECK(loop.join());
    CHECK(s.exitCode() == 0);
    m = out.messages();
    CHECK(answerTo(m, 7)["result"]["cancelled"] == true);
    std::vector<json> finished = events(m, "run_finished");
    REQUIRE(finished.size() == 1);
    CHECK(finished[0]["result"]["status"] == "cancelled");
}

TEST_CASE("agent protocol: session cancel {id} cancels only what that id names", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());
    MainLoop loop(s);
    const auto runFinished = [](const json& e) { return e.value("event", "") == "run_finished"; };

    s.receive(R"({"id":1,"method":"start_run","params":{"ms":-1}})");
    REQUIRE(out.waitFor(answers(1)));
    s.receive(R"({"id":2,"method":"echo"})");
    s.receive(R"({"id":3,"method":"run_status"})");
    REQUIRE(out.waitFor(answers(3)));

    // A quick request answered a moment ago, a run_status that reported the
    // run (cancelling it only ends a wait), and an id nobody sent: nothing
    // is cancelled, and the run goes on.
    s.receive(R"({"id":4,"method":"cancel","params":{"id":2}})");
    s.receive(R"({"id":5,"method":"cancel","params":{"id":3}})");
    s.receive(R"({"id":6,"method":"cancel","params":{"id":"nobody"}})");
    CHECK(d.cancels == 0);

    // The request being answered stops; the run does not.
    s.receive(R"({"id":7,"method":"slow"})");
    REQUIRE(waitUntil([&] { return d.slowStarted.load(); }));
    s.receive(R"({"id":8,"method":"cancel","params":{"id":7}})");
    REQUIRE(out.waitFor(answers(7)));
    CHECK(d.slowEnded);
    CHECK(d.cancels == 0);
    CHECK(d.runActive);

    // The request that started the run cancels it, once.
    s.receive(R"({"id":9,"method":"cancel","params":{"id":1}})");
    REQUIRE(out.waitFor(runFinished));
    s.receive(R"({"id":10,"method":"cancel","params":{"id":1}})");
    CHECK(d.cancels == 1);

    s.endOfInput();
    CHECK(loop.join());
    std::vector<json> m = out.messages();
    for (int id : {4, 5, 6}) {
        CAPTURE(id);
        CHECK(answerTo(m, id)["result"]["cancelled"] == false);
    }
    CHECK(answerTo(m, 7)["error"]["code"] == "cancelled");
    CHECK(answerTo(m, 8)["result"]["cancelled"] == true);
    CHECK(answerTo(m, 9)["result"]["cancelled"] == true);
    CHECK(answerTo(m, 10)["result"]["cancelled"] == false);
    std::vector<json> finished = events(m, "run_finished");
    REQUIRE(finished.size() == 1);
    CHECK(finished[0]["result"]["status"] == "cancelled");
}

TEST_CASE("agent protocol: session cancel {} cancels the run", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());
    s.receive(R"({"id":1,"method":"start_run","params":{"ms":-1}})");
    REQUIRE(s.step(0ms));
    REQUIRE(d.runActive);
    s.receive(R"({"id":2,"method":"cancel"})");
    s.endOfInput();
    REQUIRE(runToEnd(s));
    CHECK(d.cancels == 1);
    std::vector<json> m = out.messages();
    CHECK(answerTo(m, 2)["result"]["cancelled"] == true);
    std::vector<json> finished = events(m, "run_finished");
    REQUIRE(finished.size() == 1);
    CHECK(finished[0]["result"]["status"] == "cancelled");
}

TEST_CASE("agent protocol: terminate cancels and finishes without waiting", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    SessionServer s(d, out.sink(), serverOptions());
    s.receive(R"({"id":1,"method":"start_run","params":{"ms":-1}})");
    REQUIRE(s.step(0ms));
    REQUIRE(d.runActive);
    s.endOfInput();
    s.terminate();
    CHECK(runToEnd(s, 1s));
    CHECK(d.cancels >= 1);
    CHECK(s.exitCode() == 0);
}

TEST_CASE("agent protocol: a sink that throws ends the server with exit code 1", "[app][agent_protocol]") {
    FakeDispatcher d;
    SessionServer s(d, [](const std::string&) { throw std::runtime_error("broken pipe"); }, serverOptions());
    s.receive(R"({"id":1,"method":"echo"})");
    CHECK_FALSE(s.step(0ms));
    CHECK(s.exitCode() == 1);
}

TEST_CASE("agent protocol: close() waits for a receive() in progress, and receive() then does nothing", "[app][agent_protocol]") {
    for (int round = 0; round < 200; ++round) {
        FakeDispatcher d;
        Output out;
        auto s = std::make_unique<SessionServer>(d, out.sink(), serverOptions());
        std::atomic<bool> go{false};
        std::thread reader([&] {
            while (!go) std::this_thread::yield();
            for (int i = 0; i < 50; ++i) s->receive(R"({"id":1,"method":"ping"})");
        });
        go = true;
        if (round % 2) std::this_thread::yield();
        s->close();
        const std::size_t answered = out.size();
        reader.join();
        CHECK(out.size() == answered);
        s.reset();
    }
}

// --- MCP, the initialize era -----------------------------------------------------

TEST_CASE("agent protocol: MCP initialize echoes a known version, and answers an unknown one with 2025-11-25",
          "[app][agent_protocol]") {
    struct Case {
        std::string requested, negotiated;
        bool title;
    };
    for (const Case& c : {Case{"2025-06-18", "2025-06-18", true}, Case{"2025-11-25", "2025-11-25", true},
                          Case{"2025-03-26", "2025-03-26", false}, Case{"2024-11-05", "2024-11-05", false},
                          Case{"1999-01-01", "2025-11-25", true}}) {
        CAPTURE(c.requested);
        FakeDispatcher d;
        Output out;
        McpServer s(d, out.sink(), serverOptions(), mcpOptions());
        s.receive(initializeRequest(c.requested, 1));
        s.receive(R"({"jsonrpc":"2.0","method":"notifications/initialized"})");
        s.endOfInput();
        REQUIRE(runToEnd(s));
        std::vector<json> m = out.messages();
        REQUIRE(m.size() == 1);   // the notification got nothing
        json& r = m[0]["result"];
        CHECK(m[0]["jsonrpc"] == "2.0");
        CHECK(m[0]["id"] == 1);
        CHECK(r["protocolVersion"] == c.negotiated);
        CHECK(r["capabilities"] == json({{"tools", json::object()}}));
        CHECK(r["serverInfo"]["name"] == "sirius");
        CHECK(r["serverInfo"]["version"] == "9.9.9");
        CHECK(r["serverInfo"].contains("title") == c.title);
        CHECK(r["instructions"] == "Open a dataset, then run.");
    }
}

TEST_CASE("agent protocol: MCP tools/list gives each version the fields it knows", "[app][agent_protocol]") {
    struct Case {
        std::string version;
        bool annotations, title;
    };
    for (const Case& c : {Case{"2025-11-25", true, true}, Case{"2025-06-18", true, true}, Case{"2025-03-26", true, false},
                          Case{"2024-11-05", false, false}}) {
        CAPTURE(c.version);
        FakeDispatcher d;
        Output out;
        McpServer s(d, out.sink(), serverOptions(), mcpOptions());
        s.receive(initializeRequest(c.version));
        s.receive(mcpRequest(1, "tools/list"));
        s.receive(mcpRequest(2, "tools/list", {{"cursor", nullptr}}));
        s.receive(mcpRequest(3, "tools/list", {{"cursor", "page-2"}}));
        s.endOfInput();
        REQUIRE(runToEnd(s));
        std::vector<json> m = out.messages();
        json* list = find(m, answers(1));
        REQUIRE(list);
        json& tools = (*list)["result"]["tools"];
        REQUIRE(tools.size() == d.tools().size());
        CHECK_FALSE((*list)["result"].contains("nextCursor"));
        json& echo = tools[0];
        CHECK(echo["name"] == "echo");
        CHECK(echo["description"].get<std::string>().size() <= 2048);
        CHECK(echo["inputSchema"]["type"] == "object");
        CHECK_FALSE(echo.contains("outputSchema"));
        CHECK(echo.contains("annotations") == c.annotations);
        CHECK(echo.contains("title") == c.title);
        CHECK(echo.contains("_meta") == c.title);
        if (c.annotations) {
            CHECK(echo["annotations"]["readOnlyHint"] == true);
            CHECK(echo["annotations"]["destructiveHint"] == false);
            CHECK(echo["annotations"]["openWorldHint"] == false);
        }
        CHECK(find(m, answers(2)));
        CHECK(answerTo(m, 2)["result"]["tools"].size() == d.tools().size());
        REQUIRE(find(m, answers(3)));
        CHECK(answerTo(m, 3)["error"]["code"] == -32602);
    }
}

TEST_CASE("agent protocol: MCP tools/call results, tool errors and protocol errors", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    McpServer s(d, out.sink(), serverOptions(), mcpOptions());
    s.receive(initializeRequest("2025-11-25"));
    s.receive(mcpRequest(1, "tools/call", {{"name", "echo"}, {"arguments", {{"a", 1}}}}));
    s.receive(mcpRequest(2, "tools/call", {{"name", "image"}}));
    s.receive(mcpRequest(3, "tools/call", {{"name", "fail"}, {"arguments", json::object()}}));
    s.receive(mcpRequest(4, "tools/call", {{"name", "no_such_tool"}}));
    s.receive(mcpRequest(5, "tools/call", {{"name", "echo"}, {"arguments", json::array({1})}}));
    s.receive(mcpRequest(6, "tools/call", {{"arguments", json::object()}}));
    s.receive(mcpRequest(7, "tools/call", {{"name", "throws"}}));
    s.receive(mcpRequest("eight", "tools/call", {{"name", "echo"}, {"arguments", {{"warn", 1}}}}));
    s.endOfInput();
    REQUIRE(runToEnd(s));
    std::vector<json> m = out.messages();

    json& ok = answerTo(m, 1)["result"];
    CHECK(ok["isError"] == false);
    CHECK(ok["structuredContent"] == json({{"a", 1}}));
    REQUIRE(ok["content"].size() == 1);
    CHECK(ok["content"][0]["type"] == "text");
    CHECK(json::parse(ok["content"][0]["text"].get<std::string>()) == json({{"a", 1}}));

    json& image = answerTo(m, 2)["result"];
    CHECK(image["isError"] == false);
    REQUIRE(image["content"].size() == 2);
    CHECK(json::parse(image["content"][0]["text"].get<std::string>())["path"] == "C:/scratch/renders/render-0001.png");
    CHECK(image["content"][1]["type"] == "image");
    CHECK(image["content"][1]["data"] == "iVBORw0KGgo=");
    CHECK(image["content"][1]["mimeType"] == "image/png");
    for (const json& item : image["content"]) CHECK(item.value("type", "") != "resource_link");

    json& failed = answerTo(m, 3)["result"];
    CHECK(failed["isError"] == true);
    CHECK(failed["content"][0]["text"] == "unknown_step: no step 99\nhint: steps are numbered from 1");
    CHECK(failed["structuredContent"]["error"]["code"] == "unknown_step");
    CHECK(failed["structuredContent"]["error"]["data"]["step"] == 99);

    CHECK(answerTo(m, 4)["error"]["code"] == -32602);
    CHECK(answerTo(m, 5)["error"]["code"] == -32602);
    CHECK(answerTo(m, 6)["error"]["code"] == -32602);
    CHECK(answerTo(m, 7)["error"]["code"] == -32603);

    // Ids come back with their JSON type; warnings follow the result.
    json* warned = find(m, answers("eight"));
    REQUIRE(warned);
    json& content = (*warned)["result"]["content"];
    REQUIRE(content.is_array());
    REQUIRE_FALSE(content.empty());
    CHECK(content.back().value("text", "").rfind("warning: ", 0) == 0);
}

TEST_CASE("agent protocol: MCP before 2025-06-18 sends no structuredContent", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    McpServer s(d, out.sink(), serverOptions(), mcpOptions());
    s.receive(initializeRequest("2025-03-26"));
    s.receive(mcpRequest(1, "tools/call", {{"name", "echo"}, {"arguments", {{"a", 1}}}}));
    s.receive(mcpRequest(2, "tools/call", {{"name", "fail"}}));
    s.endOfInput();
    REQUIRE(runToEnd(s));
    std::vector<json> m = out.messages();
    CHECK_FALSE(answerTo(m, 1)["result"].contains("structuredContent"));
    CHECK(answerTo(m, 2)["result"]["isError"] == true);
    CHECK_FALSE(answerTo(m, 2)["result"].contains("structuredContent"));
}

TEST_CASE("agent protocol: MCP malformed messages get JSON-RPC errors", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    McpServer s(d, out.sink(), serverOptions(), mcpOptions());

    // All of these are answered on the reader thread.
    s.receive("{not json");
    s.receive(R"([{"jsonrpc":"2.0","id":1,"method":"ping"}])");
    s.receive(R"({"jsonrpc":"2.0","id":null,"method":"ping"})");
    s.receive(R"({"jsonrpc":"1.0","id":3,"method":"ping"})");
    s.receive(R"({"id":4,"method":"ping"})");
    s.receive(R"({"jsonrpc":"2.0","id":"p","method":"ping"})");
    s.receive(R"({"jsonrpc":"2.0","id":5})");
    s.receive(R"({"jsonrpc":"2.0","id":6,"result":{}})");   // a response: nothing to answer
    std::vector<json> m = out.messages();
    REQUIRE(m.size() == 7);
    CHECK(m[0]["error"]["code"] == -32700);
    CHECK_FALSE(m[0].contains("id"));
    CHECK(m[1]["error"]["code"] == -32600);
    CHECK_FALSE(m[1].contains("id"));
    CHECK(m[2]["error"]["code"] == -32600);
    CHECK_FALSE(m[2].contains("id"));
    CHECK(m[3]["error"]["code"] == -32600);
    CHECK(m[3]["id"] == 3);
    CHECK(m[4]["error"]["code"] == -32600);
    CHECK(m[4]["id"] == 4);
    CHECK(answers("p")(m[5]));
    CHECK(m[5]["result"] == json::object());
    CHECK(m[6]["error"]["code"] == -32600);
    CHECK(m[6]["id"] == 5);

    // Before initialize, without modern metadata: refused; after it, unknown
    // and unsupported methods are not found.
    s.receive(mcpRequest(10, "tools/list"));
    s.receive(initializeRequest("2025-11-25"));
    const std::vector<std::string> unsupported = {"resources/list", "prompts/list", "logging/setLevel", "no/such"};
    for (std::size_t i = 0; i < unsupported.size(); ++i) s.receive(mcpRequest(11 + static_cast<int>(i), unsupported[i]));
    s.endOfInput();
    REQUIRE(runToEnd(s));
    m = out.messages();
    CHECK(answerTo(m, 10)["error"]["code"] == -32602);
    for (int i = 11; i <= 14; ++i) CHECK(answerTo(m, i)["error"]["code"] == -32601);
}

TEST_CASE("agent protocol: MCP notifications/cancelled silences the request it names", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    McpServer s(d, out.sink(), serverOptions(), mcpOptions());
    MainLoop loop(s);
    s.receive(initializeRequest("2025-11-25", 1));
    REQUIRE(out.waitFor(answers(1)));

    s.receive(mcpRequest(7, "tools/call", {{"name", "slow"}, {"_meta", {{"progressToken", "slow"}}}}));
    REQUIRE(waitUntil([&] { return d.slowStarted.load(); }));
    // Queued behind the slow call, then cancelled: dropped unanswered.
    s.receive(mcpRequest(8, "tools/call", {{"name", "echo"}}));
    s.receive(R"({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":8,"reason":"not needed"}})");
    // Ignored: an unknown id, no id, a malformed one.
    s.receive(R"({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":99}})");
    s.receive(R"({"jsonrpc":"2.0","method":"notifications/cancelled","params":{}})");
    s.receive(R"({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":{"x":1}}})");
    CHECK(d.slowRunning);
    s.receive(R"({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":7}})");
    REQUIRE(waitUntil([&] { return d.slowEnded.load(); }));

    s.receive(mcpRequest(9, "tools/call", {{"name", "echo"}}));
    REQUIRE(out.waitFor(answers(9)));
    s.endOfInput();
    CHECK(loop.join());

    std::vector<json> m = out.messages();
    CHECK_FALSE(find(m, answers(7)));
    CHECK_FALSE(find(m, answers(8)));
    // The run was not touched: `run` cancels its own job, run_status only stops waiting.
    CHECK(d.cancels == 0);
    // Progress for the cancelled request stopped with it.
    const std::ptrdiff_t lastProgress = [&] {
        std::ptrdiff_t last = -1;
        for (std::size_t i = 0; i < m.size(); ++i)
            if (m[i].value("method", "") == "notifications/progress") last = static_cast<std::ptrdiff_t>(i);
        return last;
    }();
    CHECK(lastProgress < indexOf(m, answers(9)));
}

TEST_CASE("agent protocol: MCP never answers a cancelled request, whatever its method", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    McpServer s(d, out.sink(), serverOptions(), mcpOptions());
    MainLoop loop(s);
    s.receive(initializeRequest("2025-11-25", 1));
    REQUIRE(out.waitFor(answers(1)));

    // A tools/list held in the dispatcher while the client cancels it.
    d.holdNextTools = true;
    s.receive(mcpRequest(2, "tools/list"));
    REQUIRE(waitUntil([&] { return d.toolsHeld.load(); }));
    s.receive(R"({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":2}})");
    d.releaseTools = true;
    s.receive(mcpRequest(3, "tools/list"));
    REQUIRE(out.waitFor(answers(3)));
    s.endOfInput();
    CHECK(loop.join());

    std::vector<json> m = out.messages();
    CHECK_FALSE(find(m, answers(2)));
    CHECK(answerTo(m, 3)["result"]["tools"].size() == d.tools().size());
}

TEST_CASE("agent protocol: MCP progress is sent only with a token, strictly increasing", "[app][agent_protocol]") {
    const json fractions = json::array({0.1, 0.1, 0.05, 0.2, 0.5, 0.5, 1.0});
    const auto progressCall = [&](const json& meta) {
        json params = {{"name", "progress"}, {"arguments", {{"fractions", fractions}}}};
        if (!meta.is_null()) params["_meta"] = meta;
        return params;
    };
    const auto progressOf = [](const std::vector<json>& m) {
        std::vector<json> out;
        for (const json& n : m)
            if (n.value("method", "") == "notifications/progress") out.push_back(n.value("params", json()));
        return out;
    };

    SECTION("with a token") {
        FakeDispatcher d;
        Output out;
        McpServer s(d, out.sink(), serverOptions(), mcpOptions());
        s.receive(initializeRequest("2025-11-25"));
        s.receive(mcpRequest(1, "tools/call", progressCall({{"progressToken", "tok"}})));
        s.endOfInput();
        REQUIRE(runToEnd(s));
        std::vector<json> m = out.messages();
        std::vector<json> p = progressOf(m);
        REQUIRE(p.size() == 4);
        const double expected[] = {10, 20, 50, 100};
        for (std::size_t i = 0; i < p.size(); ++i) {
            CHECK(p[i]["progressToken"] == "tok");
            CHECK(p[i]["progress"].get<double>() == expected[i]);
            CHECK(p[i]["total"] == 100);
            CHECK(p[i]["message"] == "Step 02 \xC2\xB7 Contrast \xC2\xB7 working");
        }
        const auto isProgress = [](const json& n) { return n.value("method", "") == "notifications/progress"; };
        CHECK(indexOf(m, isProgress) < indexOf(m, answers(1)));
    }
    SECTION("an integer token, throttled") {
        FakeDispatcher d;
        Output out;
        ServerOptions o = serverOptions();
        o.progressInterval = std::chrono::hours(1);
        McpServer s(d, out.sink(), o, mcpOptions());
        s.receive(initializeRequest("2025-11-25"));
        s.receive(mcpRequest(1, "tools/call", progressCall({{"progressToken", 42}})));
        s.endOfInput();
        REQUIRE(runToEnd(s));
        std::vector<json> p = progressOf(out.messages());
        REQUIRE(p.size() == 1);
        CHECK(p[0]["progressToken"] == 42);
    }
    SECTION("2024-11-05 has no progress message") {
        FakeDispatcher d;
        Output out;
        McpServer s(d, out.sink(), serverOptions(), mcpOptions());
        s.receive(initializeRequest("2024-11-05"));
        s.receive(mcpRequest(1, "tools/call", progressCall({{"progressToken", "t"}})));
        s.endOfInput();
        REQUIRE(runToEnd(s));
        std::vector<json> p = progressOf(out.messages());
        REQUIRE_FALSE(p.empty());
        for (const json& n : p) CHECK_FALSE(n.contains("message"));
    }
    SECTION("without a token") {
        FakeDispatcher d;
        Output out;
        McpServer s(d, out.sink(), serverOptions(), mcpOptions());
        s.receive(initializeRequest("2025-11-25"));
        s.receive(mcpRequest(1, "tools/call", progressCall(nullptr)));
        s.endOfInput();
        REQUIRE(runToEnd(s));
        std::vector<json> m = out.messages();
        CHECK(progressOf(m).empty());
        CHECK(find(m, answers(1)));
    }
}

TEST_CASE("agent protocol: MCP never writes log or run_finished events, and end of input cancels a run",
          "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    McpServer s(d, out.sink(), serverOptions(), mcpOptions());
    s.receive(initializeRequest("2025-11-25"));
    s.receive(mcpRequest(1, "tools/call", {{"name", "log"}}));
    s.receive(mcpRequest(2, "tools/call", {{"name", "start_run"}, {"arguments", {{"ms", 0}}}}));
    s.receive(mcpRequest(3, "tools/call", {{"name", "start_run"}, {"arguments", {{"ms", -1}}}}));
    s.endOfInput();
    const auto start = Clock::now();
    REQUIRE(runToEnd(s));
    CHECK(Clock::now() - start < 5s);
    CHECK(d.cancels >= 1);
    CHECK(s.exitCode() == 0);
    std::vector<json> m = out.messages();
    CHECK(m.size() == 4);
    for (const json& n : m) {
        CHECK_FALSE(n.contains("event"));
        CHECK(n.value("jsonrpc", "") == "2.0");
    }
}

TEST_CASE("agent protocol: MCP end of input answers what was queued for a grace, then cancels the rest",
          "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    ServerOptions o = serverOptions();
    o.eofRunWait = 100ms;
    McpServer s(d, out.sink(), o, mcpOptions());
    s.receive(initializeRequest("2025-11-25", 0));
    s.receive(mcpRequest(1, "tools/call", {{"name", "echo"}}));
    s.receive(mcpRequest(2, "tools/call", {{"name", "start_run"}, {"arguments", {{"ms", -1}}}}));
    s.receive(mcpRequest(3, "tools/call", {{"name", "slow"}}));   // waits until it is cancelled
    s.receive(mcpRequest(4, "tools/call", {{"name", "echo"}}));
    const auto start = Clock::now();
    s.endOfInput();
    REQUIRE(runToEnd(s));
    const auto took = Clock::now() - start;
    CHECK(took >= 100ms);
    CHECK(took < 5s);
    CHECK(s.exitCode() == 0);

    // Answered in the grace: everything before the slow call.
    std::vector<json> m = out.messages();
    for (int id : {0, 1, 2}) {
        CAPTURE(id);
        CHECK(find(m, answers(id)));
    }
    // The slow call was cancelled when the grace ran out (it was not the
    // client's cancel, so it is answered), the call behind it was dropped,
    // and the run was cancelled.
    CHECK(d.slowEnded);
    CHECK(answerTo(m, 3)["result"]["isError"] == true);
    CHECK(answerTo(m, 3)["result"]["structuredContent"]["error"]["code"] == "cancelled");
    CHECK_FALSE(find(m, answers(4)));
    CHECK(d.cancels >= 1);
}

TEST_CASE("agent protocol: MCP refuses an image over maxInlineImageBytes as a too_large tool error", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    ServerOptions o = serverOptions();
    o.maxInlineImageBytes = 4;
    McpServer s(d, out.sink(), o, mcpOptions());
    s.receive(initializeRequest("2025-11-25"));
    s.receive(mcpRequest(1, "tools/call", {{"name", "image"}}));
    s.endOfInput();
    REQUIRE(runToEnd(s));
    std::vector<json> m = out.messages();
    REQUIRE(find(m, answers(1)));
    json& r = answerTo(m, 1)["result"];
    CHECK(r["isError"] == true);
    CHECK(r["content"][0]["text"].get<std::string>().rfind("too_large: ", 0) == 0);
    CHECK(r["structuredContent"]["error"]["code"] == "too_large");
    for (const json& item : r["content"]) CHECK(item.value("type", "") != "image");
}

TEST_CASE("agent protocol: maxInlineImageBytes limits the images of one answer together", "[app][agent_protocol]") {
    // Each image is 8 bytes, under the limit alone; two are over it.
    ServerOptions o = serverOptions();
    o.maxInlineImageBytes = 12;
    {
        FakeDispatcher d;
        Output out;
        SessionServer s(d, out.sink(), o);
        s.receive(R"({"id":1,"method":"image","params":{"count":2,"inline":true}})");
        s.receive(R"({"id":2,"method":"image","params":{"count":2}})");
        s.receive(R"({"id":3,"method":"image","inline":true})");
        s.endOfInput();
        REQUIRE(runToEnd(s));
        std::vector<json> m = out.messages();
        REQUIRE(find(m, answers(1)));
        CHECK(answerTo(m, 1)["ok"] == false);
        CHECK(answerTo(m, 1)["error"]["code"] == "too_large");
        CHECK(answerTo(m, 1)["error"]["data"]["bytes"] == 16);
        // Without inline nothing large goes out, so there is no limit.
        REQUIRE(find(m, answers(2)));
        CHECK(answerTo(m, 2)["result"]["images"].size() == 2);
        // "inline" beside the params works as well as in them.
        REQUIRE(find(m, answers(3)));
        CHECK(answerTo(m, 3)["result"]["image"]["base64"] == "iVBORw0KGgo=");
        CHECK_FALSE(d.lastArgs.contains("inline"));
    }
    {
        FakeDispatcher d;
        Output out;
        McpServer s(d, out.sink(), o, mcpOptions());
        s.receive(initializeRequest("2025-11-25"));
        s.receive(mcpRequest(1, "tools/call", {{"name", "image"}, {"arguments", {{"count", 2}}}}));
        s.receive(mcpRequest(2, "tools/call", {{"name", "image"}}));
        s.endOfInput();
        REQUIRE(runToEnd(s));
        std::vector<json> m = out.messages();
        REQUIRE(find(m, answers(1)));
        CHECK(answerTo(m, 1)["result"]["isError"] == true);
        CHECK(answerTo(m, 1)["result"]["structuredContent"]["error"]["data"]["bytes"] == 16);
        REQUIRE(find(m, answers(2)));
        CHECK(answerTo(m, 2)["result"]["isError"] == false);
    }
}

// --- MCP, 2026-07-28 -----------------------------------------------------------------

TEST_CASE("agent protocol: MCP server/discover and per-request metadata", "[app][agent_protocol]") {
    FakeDispatcher d;
    Output out;
    McpServer s(d, out.sink(), serverOptions(), mcpOptions());
    s.receive(mcpRequest(1, "server/discover", {{"_meta", modernMeta()}}));
    s.receive(mcpRequest(2, "tools/list", {{"_meta", modernMeta()}}));
    s.receive(mcpRequest(3, "tools/call", {{"name", "echo"}, {"arguments", {{"a", 1}}}, {"_meta", modernMeta()}}));
    s.receive(mcpRequest(4, "tools/call", {{"name", "echo"}, {"_meta", modernMeta("1900-01-01")}}));
    const json noCapabilities = {{"io.modelcontextprotocol/protocolVersion", "2026-07-28"}};
    s.receive(mcpRequest(5, "tools/call", {{"name", "echo"}, {"_meta", noCapabilities}}));
    s.receive(mcpRequest(6, "tools/call", {{"name", "echo"}}));
    s.receive(mcpRequest(7, "subscriptions/listen", {{"_meta", modernMeta()}}));
    s.receive(mcpRequest(8, "ping"));
    s.endOfInput();
    REQUIRE(runToEnd(s));
    std::vector<json> m = out.messages();
    const auto serverInfoOf = [](json& result) { return result["_meta"]["io.modelcontextprotocol/serverInfo"]; };

    json& discover = answerTo(m, 1)["result"];
    CHECK(discover["resultType"] == "complete");
    json& versions = discover["supportedVersions"];
    for (const char* v : {"2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"})
        CHECK(std::find(versions.begin(), versions.end(), json(v)) != versions.end());
    CHECK(discover["capabilities"] == json({{"tools", json::object()}}));
    CHECK(discover["instructions"] == "Open a dataset, then run.");
    CHECK(discover["ttlMs"] == 3600000);
    CHECK(discover["cacheScope"] == "public");
    CHECK(serverInfoOf(discover)["name"] == "sirius");
    CHECK(serverInfoOf(discover)["version"] == "9.9.9");

    json& list = answerTo(m, 2)["result"];
    CHECK(list["resultType"] == "complete");
    CHECK(list["ttlMs"] == 3600000);
    CHECK(list["cacheScope"] == "public");
    CHECK(list["tools"][0].contains("title"));
    CHECK(list["tools"][0].contains("annotations"));

    json& call = answerTo(m, 3)["result"];
    CHECK(call["resultType"] == "complete");
    CHECK(serverInfoOf(call)["name"] == "sirius");
    CHECK(call["structuredContent"] == json({{"a", 1}}));
    CHECK(call["isError"] == false);

    json& unsupported = answerTo(m, 4)["error"];
    CHECK(unsupported["code"] == -32022);
    CHECK(unsupported["data"]["requested"] == "1900-01-01");
    CHECK(unsupported["data"]["supported"] == json::array({"2026-07-28"}));

    CHECK(answerTo(m, 5)["error"]["code"] == -32602);   // no client capabilities
    CHECK(answerTo(m, 6)["error"]["code"] == -32602);   // neither metadata nor initialize
    CHECK(answerTo(m, 7)["error"]["code"] == -32601);
    CHECK(answerTo(m, 8)["result"] == json::object());
}
