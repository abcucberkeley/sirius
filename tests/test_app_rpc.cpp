// The worker protocol: frame encoding, a scripted worker on the loopback
// transport answering hello / run / cancel, progress streaming and error
// propagation. The Python worker implements the same bytes; its own tests
// live in app/python/tests.

// requireOperation returns a reference to a registry-owned object; GCC 13's
// -Wdangling-reference cannot see that and flags the binding.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <atomic>
#include <thread>

#include "core/errors.hpp"
#include "core/rpc.hpp"

using namespace sirius;
using namespace sirius::app;
using json = nlohmann::json;

namespace {

    // Minimal worker: answers on its own thread until the transport closes.
    struct ScriptedWorker {
        std::unique_ptr<rpc::Transport> t;
        std::thread thread;
        std::atomic<bool> stop{false};
        std::string expectedToken;
        bool slow = false;
        // What this worker claims in its hello reply; -1 answers without the
        // field at all, as a worker predating the handshake does.
        int protocolVersion = rpc::kProtocolVersion;
        std::atomic<int> sawClientVersion{-1};   // the version the client sent in its hello

        explicit ScriptedWorker(std::unique_ptr<rpc::Transport> transport, std::string token = {}, bool slowRun = false,
                                int version = rpc::kProtocolVersion)
            : t(std::move(transport)), expectedToken(std::move(token)), slow(slowRun), protocolVersion(version) {
            thread = std::thread([this] { loop(); });
        }
        ~ScriptedWorker() {
            stop = true;
            t->close();
            thread.join();
        }
        void send(const json& h, const std::vector<rpc::TensorRef>& tensors = {}) { t->send(rpc::encodeFrame(h, tensors)); }
        void loop() {
            std::vector<std::byte> buf;
            std::atomic<bool> cancelled{false};
            try {
                while (!stop) {
                    auto m = rpc::decodeFrame(buf);
                    if (!m) {
                        t->receive(buf, std::chrono::milliseconds(50));
                        continue;
                    }
                    const json& h = m->header;
                    const std::uint64_t id = h.value("id", 0ull);
                    const std::string method = h.value("method", "");
                    if (method == "hello") {
                        sawClientVersion = h.contains("params") ? h["params"].value("protocol_version", -1) : -1;
                        if (!expectedToken.empty() && h.value("token", "") != expectedToken) {
                            send({{"id", id}, {"type", "error"}, {"message", "bad token"}});
                            continue;
                        }
                        json caps = {{"version", "test"}, {"methods", {"run:torch_segment", "model_info"}}, {"cuda", false}, {"device", "cpu"}, {"hostname", "loop"}};
                        if (protocolVersion >= 0) caps["protocol_version"] = protocolVersion;
                        send({{"id", id}, {"type", "result"}, {"result", caps}});
                    } else if (method == "cancel") {
                        cancelled = true;
                    } else if (method == "run") {
                        const std::string kind = h["params"].value("kind", "");
                        if (kind == "fail") {
                            send({{"id", id}, {"type", "error"}, {"message", "kaboom"}});
                            continue;
                        }
                        REQUIRE(m->tensors.size() == 1);
                        const rpc::Tensor& in = m->tensors[0];
                        cancelled = false;
                        for (int i = 0; i < (slow ? 40 : 3); ++i) {
                            // keep servicing cancel requests while "working"
                            auto c = rpc::decodeFrame(buf);
                            if (!c) t->receive(buf, std::chrono::milliseconds(slow ? 25 : 1));
                            else if (c->header.value("method", "") == "cancel") cancelled = true;
                            if (cancelled) break;
                            send({{"id", id}, {"type", "progress"}, {"fraction", (i + 1) / 3.0}, {"message", "tile " + std::to_string(i)}});
                        }
                        if (cancelled) {
                            send({{"id", id}, {"type", "error"}, {"message", "cancelled"}});
                            continue;
                        }
                        std::vector<float> out(static_cast<std::size_t>(in.numel()));
                        const float* src = in.asFloat32();
                        for (std::size_t i = 0; i < out.size(); ++i) out[i] = src[i] * 2.0f;
                        rpc::TensorRef ref{"prob", "float32", in.shape, out.data(), out.size() * sizeof(float)};
                        send({{"id", id}, {"type", "result"}, {"result", {{"channels", 1}}}}, {ref});
                    } else {
                        send({{"id", id}, {"type", "error"}, {"message", "unknown method " + method}});
                    }
                }
            } catch (const std::exception&) {
                // the client closed: done
            }
        }
    };

} // namespace

TEST_CASE("rpc frames round trip with tensors", "[app][rpc]") {
    std::vector<float> a{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<std::uint32_t> b{7u, 8u};
    std::vector<rpc::TensorRef> refs{{"a", "float32", {2, 3}, a.data(), a.size() * sizeof(float)},
                                     {"b", "uint32", {2}, b.data(), b.size() * sizeof(std::uint32_t)}};
    const json header = {{"id", 5}, {"type", "request"}, {"method", "run"}, {"params", {{"kind", "x"}}}};
    std::vector<std::byte> bytes = rpc::encodeFrame(header, refs);
    CHECK(bytes.size() > 4 + 8 + 24 + 8);

    SECTION("a partial buffer yields nothing and keeps its bytes") {
        std::vector<std::byte> part(bytes.begin(), bytes.begin() + 10);
        CHECK_FALSE(rpc::decodeFrame(part));
        CHECK(part.size() == 10);
    }
    SECTION("two frames back to back decode one at a time") {
        std::vector<std::byte> two = bytes;
        two.insert(two.end(), bytes.begin(), bytes.end());
        auto m1 = rpc::decodeFrame(two);
        REQUIRE(m1);
        CHECK(m1->header["method"] == "run");
        REQUIRE(m1->tensors.size() == 2);
        CHECK(m1->tensors[0].shape == std::vector<Index>{2, 3});
        CHECK(m1->tensors[0].asFloat32()[5] == 6.f);
        CHECK(m1->tensors[1].asUInt32()[1] == 8u);
        CHECK_THROWS(m1->tensors[1].asFloat32());
        auto m2 = rpc::decodeFrame(two);
        REQUIRE(m2);
        CHECK(two.empty());
    }
    SECTION("size mismatches are rejected") {
        std::vector<rpc::TensorRef> bad{{"a", "float32", {2, 2}, a.data(), a.size() * sizeof(float)}};
        CHECK_THROWS(rpc::encodeFrame(header, bad));
        std::vector<std::byte> garbage(20, std::byte{0xff});
        CHECK_THROWS(rpc::decodeFrame(garbage));
    }
}

// Every length in a frame header comes from the peer. Each case below would,
// without the caps and the checked arithmetic in decodeFrame, wrap an
// intermediate and let the decoder index past the end of the buffer.
TEST_CASE("rpc frames with hostile lengths are refused, not wrapped", "[app][rpc]") {
    auto put32 = [](std::vector<std::byte>& out, std::uint32_t v) {
        for (int i = 3; i >= 0; --i) out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
    };
    auto put64 = [](std::vector<std::byte>& out, std::uint64_t v) {
        for (int i = 7; i >= 0; --i) out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
    };
    auto frame = [&](const json& header, std::uint64_t payloadLen) {
        const std::string h = header.dump();
        std::vector<std::byte> out;
        put32(out, static_cast<std::uint32_t>(h.size()));
        for (char c : h) out.push_back(static_cast<std::byte>(c));
        put64(out, payloadLen);
        return out;
    };

    SECTION("a payload length that would wrap the frame total is rejected") {
        // 4 + hlen + 8 + plen wraps to a small number for a plen near 2^64,
        // so "is the whole frame here yet" would say yes on a 30-byte buffer.
        std::vector<std::byte> f = frame(json{{"id", 1}}, ~0ull - 16);
        CHECK_THROWS(rpc::decodeFrame(f));
    }
    SECTION("an implausible payload length is rejected before allocation") {
        std::vector<std::byte> f = frame(json{{"id", 1}}, 1ull << 45);
        CHECK_THROWS(rpc::decodeFrame(f));
    }
    SECTION("a tensor shape whose product overflows is rejected") {
        const json header{{"id", 1},
                          {"tensors", json::array({json{{"name", "t"},
                                                        {"dtype", "float32"},
                                                        {"shape", json::array({1 << 20, 1 << 20, 1 << 20, 1 << 20})},
                                                        {"offset", 0},
                                                        {"nbytes", 4}}})}};
        std::vector<std::byte> f = frame(header, 4);
        for (int i = 0; i < 4; ++i) f.push_back(std::byte{0});
        CHECK_THROWS(rpc::decodeFrame(f));
    }
    SECTION("a framing failure is a ProtocolError, and still a runtime_error") {
        // The typed failure is what lets a caller tell "the connection to the
        // worker broke" from "the step the worker ran failed", which used to
        // mean noticing that one message starts with "rpc: ".
        std::vector<std::byte> f = frame(json{{"id", 1}}, 1ull << 45);
        CHECK_THROWS_AS(rpc::decodeFrame(f), ProtocolError);
        CHECK_THROWS_AS(rpc::decodeFrame(f), sirius::SiriusError);
        CHECK_THROWS_AS(rpc::decodeFrame(f), std::runtime_error);
        CHECK_THROWS_AS(rpc::decodeFrame(f), std::exception);
        CHECK_THROWS_WITH(rpc::decodeFrame(f), Catch::Matchers::StartsWith("rpc: "));   // the message did not move
    }
    SECTION("a tensor whose offset plus size wraps is rejected") {
        const json header{{"id", 1},
                          {"tensors", json::array({json{{"name", "t"},
                                                        {"dtype", "float32"},
                                                        {"shape", json::array({1})},
                                                        {"offset", ~0ull - 2},
                                                        {"nbytes", 8}}})}};
        std::vector<std::byte> f = frame(header, 4);
        for (int i = 0; i < 4; ++i) f.push_back(std::byte{0});
        CHECK_THROWS(rpc::decodeFrame(f));
    }
}

TEST_CASE("RemoteWorker talks to a scripted worker over the loopback", "[app][rpc]") {
    auto [client, server] = rpc::loopbackPair();
    ScriptedWorker worker(std::move(server), "secret");

    SECTION("a wrong token is refused") {
        CHECK_THROWS_WITH(RemoteWorker(std::move(client), "nope"), Catch::Matchers::ContainsSubstring("bad token"));
    }
    SECTION("hello, run with progress, error") {
        RemoteWorker rw(std::move(client), "secret");
        CHECK(rw.capabilities().version == "test");
        CHECK(rw.supports("torch_segment"));
        CHECK_FALSE(rw.supports("sim"));
        std::vector<float> in{1.f, 2.f, 3.f, 4.f};
        std::vector<double> fractions;
        WorkerResult r = rw.call("run", {{"kind", "torch_segment"}}, {{"input", "float32", {2, 2}, in.data(), 16}},
                                 [&](double f, const std::string&) { fractions.push_back(f); });
        CHECK(fractions.size() == 3);
        REQUIRE(r.tensors.size() == 1);
        CHECK(r.tensors[0].name == "prob");
        CHECK(r.tensors[0].asFloat32()[3] == 8.f);
        CHECK(r.result["channels"] == 1);
        CHECK_THROWS_WITH(rw.call("run", {{"kind", "fail"}}, {{"input", "float32", {1}, in.data(), 4}}),
                          Catch::Matchers::ContainsSubstring("kaboom"));
        rw.close();
        CHECK_FALSE(rw.isOpen());
    }
}

// The version handshake: "hello" carries rpc::kProtocolVersion both ways and
// the client refuses anything else, naming both numbers so the user knows
// which end to update. The Python worker enforces the same rule
// (sirius_worker/protocol.py: PROTOCOL_VERSION).
TEST_CASE("RemoteWorker refuses a worker speaking another protocol version", "[app][rpc]") {
    SECTION("a matching version connects and is reported") {
        auto [client, server] = rpc::loopbackPair();
        ScriptedWorker worker(std::move(server), "secret");
        RemoteWorker rw(std::move(client), "secret");
        CHECK(rw.capabilities().protocolVersion == rpc::kProtocolVersion);
        CHECK(worker.sawClientVersion.load() == rpc::kProtocolVersion);   // the client sends its own version too
    }
    SECTION("a newer worker names both versions and says to update the application") {
        auto [client, server] = rpc::loopbackPair();
        ScriptedWorker worker(std::move(server), "secret", false, rpc::kProtocolVersion + 6);
        CHECK_THROWS_WITH(RemoteWorker(std::move(client), "secret"),
                          Catch::Matchers::ContainsSubstring("version " + std::to_string(rpc::kProtocolVersion)) &&
                              Catch::Matchers::ContainsSubstring("version " + std::to_string(rpc::kProtocolVersion + 6)) &&
                              Catch::Matchers::ContainsSubstring("update SIRIUS"));
    }
    SECTION("a worker predating the handshake counts as version 0 and must be updated") {
        auto [client, server] = rpc::loopbackPair();
        ScriptedWorker worker(std::move(server), "secret", false, -1);   // no protocol_version field at all
        CHECK_THROWS_WITH(RemoteWorker(std::move(client), "secret"),
                          Catch::Matchers::ContainsSubstring("version 0") &&
                              Catch::Matchers::ContainsSubstring("update sirius_worker"));
    }
}

TEST_CASE("RemoteWorker cancels a slow request", "[app][rpc]") {
    auto [client, server] = rpc::loopbackPair();
    ScriptedWorker worker(std::move(server), "", true);
    RemoteWorker rw(std::move(client));
    std::vector<float> in{1.f};
    std::atomic<bool> cancel{false};
    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        cancel = true;
    });
    CHECK_THROWS_WITH(rw.call("run", {{"kind", "torch_segment"}}, {{"input", "float32", {1}, in.data(), 4}}, {},
                              [&] { return cancel.load(); }),
                      Catch::Matchers::ContainsSubstring("cancelled"));
    canceller.join();
}

TEST_CASE("connectTcp reports an unreachable port", "[app][rpc]") {
    CHECK_THROWS(rpc::connectTcp("127.0.0.1", 1, std::chrono::milliseconds(500)));
    (void)workerScriptPath("/definitely/not/here");   // must not throw
}

// --- the real Python worker ------------------------------------------------------
// Runs only when SIRIUS_PYTHON names an interpreter with numpy (the conda one
// on the dev machine); CI has no worker and skips.

#ifndef _WIN32
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>

#include "temp_path.hpp"

TEST_CASE("the bundled Python worker answers hello and runs a numpy step", "[app][rpc][worker]") {
    const char* python = std::getenv("SIRIUS_PYTHON");
    if (!python || !*python) SKIP("SIRIUS_PYTHON is not set");
    const std::string dir = workerScriptPath();
    if (dir.empty()) SKIP("sirius_worker not found");
    const std::string cmd = std::string("cd '") + dir + "' && exec '" + python +
                            "' -m sirius_worker --host 127.0.0.1 --port 0 --token abc --device cpu 2>/dev/null";
    FILE* pipe = ::popen(cmd.c_str(), "r");
    REQUIRE(pipe);
    char line[512] = {0};
    REQUIRE(std::fgets(line, sizeof line, pipe));
    const json hello = json::parse(line);
    const int port = hello.value("port", 0);
    REQUIRE(port > 0);

    auto worker = RemoteWorker::connect("127.0.0.1", port, "abc");
    CHECK(worker->supports("torch_segment"));
    CHECK_FALSE(worker->capabilities().hostname.empty());
    // mean over t of a (c, t, z, y, x) array through the "einsum" kind
    std::vector<float> in(2 * 3 * 1 * 2 * 2);
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);
    WorkerResult r = worker->call("run", {{"kind", "einsum"}, {"params", {{"axes", "czyx"}, {"reduction", "mean"}}}},
                                  {{"input", "float32", {2, 3, 1, 2, 2}, in.data(), in.size() * sizeof(float)}});
    REQUIRE_FALSE(r.tensors.empty());
    const rpc::Tensor& out = r.tensors.front();
    CHECK(out.shape == std::vector<Index>{2, 1, 1, 2, 2});
    // element (c0, y0, x0): mean of 0, 4, 8 = 4
    CHECK(out.asFloat32()[0] == 4.0f);
    CHECK_THROWS(worker->call("run", {{"kind", "no_such_kind"}}));
    (void)worker->call("shutdown", json::object());
    worker->close();
    ::pclose(pipe);
}
#endif

#ifndef _WIN32
#include "core/help_pages.hpp"
#include "core/ops/builtin.hpp"
#include "core/ops/plugin.hpp"
#include "core/array_source.hpp"

// End to end: a TorchScript model scripted by the worker's own Python, the
// worker launched as the app does, and the segmentation operation run
// against it. Needs torch in SIRIUS_PYTHON; skips otherwise.
TEST_CASE("the segmentation step runs a TorchScript model through the worker", "[app][rpc][worker][seg]") {
    const char* python = std::getenv("SIRIUS_PYTHON");
    if (!python || !*python) SKIP("SIRIUS_PYTHON is not set");
    const std::string dir = workerScriptPath();
    if (dir.empty()) SKIP("sirius_worker not found");
    registerBuiltinOperations();

    // foreground probability = sigmoid of the intensity around 0.5; boundary channel = 0
    test::TempFile model("segmodel", ".pt");
    const std::string script =
        "import sys\n"
        "try:\n    import torch\nexcept ImportError:\n    sys.exit(3)\n"
        "class M(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        fg = torch.sigmoid((x - 0.5) * 20.0)\n"
        "        return torch.cat([fg, torch.zeros_like(x)], 1)\n"
        "torch.jit.script(M()).save(" +
        json(model.str).dump() + ")\n";
    test::TempFile scriptFile("segmodel", ".py");
    {
        std::ofstream out(scriptFile.path);
        out << script;
    }
    const int rc = std::system((std::string("'") + python + "' '" + scriptFile.str + "' >/dev/null 2>&1").c_str());
    if (WEXITSTATUS(rc) == 3) SKIP("torch is not importable in SIRIUS_PYTHON");
    REQUIRE(WEXITSTATUS(rc) == 0);

    const std::string cmd = std::string("cd '") + dir + "' && exec '" + python +
                            "' -m sirius_worker --host 127.0.0.1 --port 0 --token seg --device cpu 2>/dev/null";
    FILE* pipe = ::popen(cmd.c_str(), "r");
    REQUIRE(pipe);
    char line[512] = {0};
    REQUIRE(std::fgets(line, sizeof line, pipe));
    const int port = json::parse(line).value("port", 0);
    REQUIRE(port > 0);
    auto worker = RemoteWorker::connect("127.0.0.1", port, "seg");

    // two bright cubes in a (1, 1, 8, 32, 32) volume
    const Dims5 dims{1, 1, 8, 32, 32};
    auto array = std::make_shared<Array5>(Array5::zeros(dims));
    for (Index z = 1; z < 6; ++z)
        for (Index y = 2; y < 9; ++y)
            for (Index x = 2; x < 9; ++x) {
                array->at(0, 0, z, y, x) = 1.0f;
                array->at(0, 0, z, y + 15, x + 15) = 1.0f;
            }
    DatasetMeta meta;
    meta.dims = dims;
    meta.normalizeChannels();
    StepInput in{meta, array, nullptr, nullptr};

    const Operation& seg = requireOperation("seg");
    ParamSet p = seg.defaults();
    p.set("model", model.str);
    p.set("post", std::string("Connected components"));
    p.set("tile", std::vector<double>{8, 32, 32});
    p.set("overlap", std::int64_t{0});
    p.set("min_voxels", std::int64_t{5});
    StepContext ctx;
    ctx.remote = worker.get();
    std::vector<double> progress;
    ctx.progress = [&](double f, const std::string&) { progress.push_back(f); };
    const StepOutput out = seg.run(in, p, ctx);
    REQUIRE(out.labels);
    CHECK(out.labels->stats().size() == 2);
    CHECK(out.labels->at(0, 3, 5, 5) != 0);
    CHECK(out.labels->at(0, 3, 20, 20) != 0);
    CHECK(out.labels->at(0, 3, 5, 5) != out.labels->at(0, 3, 20, 20));
    CHECK(out.labels->at(0, 0, 0, 0) == 0);
    CHECK_FALSE(progress.empty());
    CHECK(out.diagnostics.kind == DiagnosticsKind::Segment);

    (void)worker->call("shutdown", json::object());
    worker->close();
    ::pclose(pipe);
}
#endif

#ifndef _WIN32
TEST_CASE("plugins from the worker become operations and run", "[app][rpc][worker][plugin]") {
    const char* python = std::getenv("SIRIUS_PYTHON");
    if (!python || !*python) SKIP("SIRIUS_PYTHON is not set");
    const std::string dir = workerScriptPath();
    if (dir.empty()) SKIP("sirius_worker not found");
    registerBuiltinOperations();

    // a plugin directory of our own: one good file, one broken, one colliding with a built-in
    const std::filesystem::path pdir = std::filesystem::temp_directory_path() / ("sirius-plugins-" + std::to_string(::getpid()));
    std::filesystem::create_directories(pdir);
    {
        std::ofstream good(pdir / "double_it.py");
        good << "STEP = {'kind': 'double_it', 'name': 'Double', 'group': 'Intensity',\n"
                "        'params': [{'key': 'factor', 'type': 'double', 'default': 2.0, 'min': 0, 'max': 10}],\n"
                "        'separable_over_t': True, 'help': '# Double\\n\\nMultiplies by *factor*.'}\n"
                "def run(data, params, meta, ctx):\n"
                "    ctx.progress(0.5, 'half')\n"
                "    return data * params['factor'], {'facts': {'factor': str(params['factor'])}}\n";
        std::ofstream bad(pdir / "broken.py");
        bad << "STEP = {'kind': 'broken', 'params': [{'key': 'x', 'type': 'nope'}]}\ndef run(d, p, m, c): return d\n";
        std::ofstream clash(pdir / "clash.py");
        clash << "STEP = {'kind': 'contrast'}\ndef run(d, p, m, c): return d\n";
    }
    const std::string cmd = std::string("cd '") + dir + "' && SIRIUS_PLUGIN_DIRS='" + pdir.string() + "' exec '" + python +
                            "' -m sirius_worker --host 127.0.0.1 --port 0 --token plug --device cpu 2>/dev/null";
    FILE* pipe = ::popen(cmd.c_str(), "r");
    REQUIRE(pipe);
    char line[512] = {0};
    REQUIRE(std::fgets(line, sizeof line, pipe));
    const int port = json::parse(line).value("port", 0);
    REQUIRE(port > 0);
    auto worker = RemoteWorker::connect("127.0.0.1", port, "plug");
    CHECK(worker->supports("plugin"));

    const PluginLoadResult loaded = registerPluginOperations(*worker, false);
    CHECK(std::find(loaded.kinds.begin(), loaded.kinds.end(), "double_it") != loaded.kinds.end());
    CHECK(std::find(loaded.kinds.begin(), loaded.kinds.end(), "contrast") == loaded.kinds.end());
    CHECK(loaded.errors.size() == 2);   // broken.py and clash.py
    const Operation* op = findOperation("double_it");
    REQUIRE(op);
    CHECK(op->info().plugin);
    CHECK(op->info().group == "User");          // every plugin lists under the User section
    CHECK(op->info().kindLabel == "INTENSITY");  // the declared group survives as the row's label
    CHECK(op->info().separableOverT);
    CHECK(op->info().params.size() == 1);
    CHECK(op->info().params[0].max == 10.0);
    CHECK(loadHelpPage("double_it").title == "Double");

    const Dims5 dims{1, 2, 2, 4, 4};
    auto array = std::make_shared<Array5>(Array5::filled(dims, 1.5f));
    DatasetMeta meta;
    meta.dims = dims;
    meta.normalizeChannels();
    ParamSet p = op->defaults();
    p.set("factor", 3.0);
    StepContext ctx;
    ctx.remote = worker.get();
    std::vector<std::string> messages;
    ctx.progress = [&](double, const std::string& m) { if (!m.empty()) messages.push_back(m); };
    const StepOutput out = op->run(StepInput{meta, array, nullptr, nullptr}, p, ctx);
    REQUIRE(out.array);
    CHECK(out.array->dims() == dims);
    CHECK(out.array->at(0, 1, 1, 2, 3) == 4.5f);
    CHECK(out.diagnostics.facts.front().value == "3.0");
    CHECK(std::find(messages.begin(), messages.end(), "half") != messages.end());
    // without a worker the step explains what to do
    StepContext none;
    CHECK_THROWS_WITH(op->run(StepInput{meta, array, nullptr, nullptr}, p, none), Catch::Matchers::ContainsSubstring("worker"));

    (void)worker->call("shutdown", json::object());
    worker->close();
    ::pclose(pipe);
    std::filesystem::remove_all(pdir);
}
#endif
