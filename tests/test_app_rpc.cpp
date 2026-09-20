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
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>

#include "core/app_paths.hpp"
#include "core/cancel.hpp"
#include "core/errors.hpp"
#include "core/ops/builtin.hpp"
#include "core/rpc.hpp"

#include <nlohmann/json.hpp>

#include "temp_path.hpp"

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
        std::mutex sentMutex;
        json foundationParams;   // what the last "foundation" run was sent

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
                        json caps = {{"version", "test"}, {"methods", {"run:torch_segment", "run:foundation", "model_info"}}, {"cuda", false}, {"device", "cpu"}, {"hostname", "loop"}};
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
                        if (kind == "fail_noid") {
                            // what the Python worker sends for a frame it could
                            // not attribute to a request (a protocol error)
                            send({{"id", nullptr}, {"type", "error"}, {"message", "protocol error: bad frame"}});
                            continue;
                        }
                        if (kind == "hang") {
                            // a stuck worker: never answers, never reads its cancel
                            while (!stop) {
                                auto c = rpc::decodeFrame(buf);
                                if (!c) t->receive(buf, std::chrono::milliseconds(25));
                            }
                            continue;
                        }
                        if (kind == "foundation") {
                            // the foundation model's reply: (t, z, y, x) uint32
                            // labels holding two one-voxel objects per frame,
                            // which is what a Detect run returns, a confidence
                            // map, and the facts the step reports
                            {
                                std::lock_guard<std::mutex> lock(sentMutex);
                                foundationParams = h["params"].value("params", json::object());
                            }
                            const std::vector<Index> s = m->tensors.at(0).shape;   // (c, t, z, y, x)
                            const Index volume = s.at(2) * s.at(3) * s.at(4);
                            std::vector<std::uint32_t> labels(static_cast<std::size_t>(s[1] * volume), 0u);
                            std::vector<float> confidence(labels.size(), 0.9f);
                            for (Index t = 0; t < s[1]; ++t) {
                                labels[static_cast<std::size_t>(t * volume + (s[3] + 1) * s[4] + 1)] = 1u;   // (1, 1, 1)
                                labels[static_cast<std::size_t>((t + 1) * volume - 1)] = 2u;   // the last voxel
                            }
                            const std::vector<Index> shape{s[1], s[2], s[3], s[4]};
                            rpc::TensorRef l{"labels", "uint32", shape, labels.data(), labels.size() * sizeof(std::uint32_t)};
                            rpc::TensorRef c{"confidence", "float32", shape, confidence.data(), confidence.size() * sizeof(float)};
                            send({{"id", id},
                                  {"type", "result"},
                                  {"result", {{"model", "stub"}, {"threshold", 0.5}, {"min_separation_um", 1.0}, {"objects", 2 * s[1]}, {"tracks", 2}, {"divisions", 1}}}},
                                 {l, c});
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
    // Little-endian, as rpc.cpp writes and reads them. These built the lengths
    // the other way round, so every frame below was rejected at the *header*
    // length -- 8 read as 0x08000000, 128 MiB, over the 64 MiB cap -- and no
    // section ever reached the check it is named after. CHECK_THROWS was
    // satisfied either way. The matchers below are what keep that from
    // happening again: each one names the check it means to exercise.
    auto put32 = [](std::vector<std::byte>& out, std::uint32_t v) {
        for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
    };
    auto put64 = [](std::vector<std::byte>& out, std::uint64_t v) {
        for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
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
        CHECK_THROWS_WITH(rpc::decodeFrame(f), Catch::Matchers::ContainsSubstring("payload of"));
    }
    SECTION("an implausible payload length is rejected before allocation") {
        // above kMaxPayloadBytes (32 GiB), and nothing has been allocated yet
        std::vector<std::byte> f = frame(json{{"id", 1}}, 64ull << 30);
        CHECK_THROWS_WITH(rpc::decodeFrame(f), Catch::Matchers::ContainsSubstring("payload of"));
    }
    SECTION("an implausible header length is rejected before the header is read") {
        std::vector<std::byte> f;
        put32(f, 128u << 20);   // 128 MiB of JSON, over the 64 MiB cap
        CHECK_THROWS_WITH(rpc::decodeFrame(f), Catch::Matchers::ContainsSubstring("header of"));
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
        CHECK_THROWS_WITH(rpc::decodeFrame(f), Catch::Matchers::ContainsSubstring("tensor"));
    }
    SECTION("a framing failure is a ProtocolError, and still a runtime_error") {
        // The typed failure is what lets a caller tell "the connection to the
        // worker broke" from "the step the worker ran failed", which used to
        // mean noticing that one message starts with "rpc: ".
        std::vector<std::byte> f = frame(json{{"id", 1}}, 64ull << 30);
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
        CHECK_THROWS_WITH(rpc::decodeFrame(f), Catch::Matchers::ContainsSubstring("tensor"));
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
    // the worker's own "cancelled" answer to our cancel is a cancellation,
    // typed as one, not a step failure that happens to say "cancelled"
    CHECK_THROWS_AS(rw.call("run", {{"kind", "torch_segment"}}, {{"input", "float32", {1}, in.data(), 4}}, {},
                            [&] { return cancel.load(); }),
                    CancelledError);
    canceller.join();
}

TEST_CASE("An error the worker cannot attribute to a request still carries its message", "[app][rpc]") {
    auto [client, server] = rpc::loopbackPair();
    ScriptedWorker worker(std::move(server));
    RemoteWorker rw(std::move(client));
    std::vector<float> in{1.f};
    // before: h.value("id", uint64) on a null id threw a JSON type error
    CHECK_THROWS_WITH(rw.call("run", {{"kind", "fail_noid"}}, {{"input", "float32", {1}, in.data(), 4}}),
                      Catch::Matchers::ContainsSubstring("bad frame"));
}

TEST_CASE("A worker that ignores a cancel is given up after the grace period", "[app][rpc][cancel]") {
    auto [client, server] = rpc::loopbackPair();
    ScriptedWorker worker(std::move(server));
    RemoteWorker rw(std::move(client));
    rw.setCancelGrace(std::chrono::milliseconds(300));
    std::vector<float> in{1.f};
    std::atomic<bool> cancel{false};
    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cancel = true;
    });
    const auto t0 = std::chrono::steady_clock::now();
    CHECK_THROWS_AS(rw.call("run", {{"kind", "hang"}}, {{"input", "float32", {1}, in.data(), 4}}, {},
                            [&] { return cancel.load(); }),
                    CancelledError);
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
    CHECK_FALSE(rw.isOpen());   // the connection is done: nothing waits on it any more
    canceller.join();
}

TEST_CASE("the foundation step keeps the labels the worker returns and reports its run", "[app][rpc][foundation]") {
    registerBuiltinOperations();
    auto [client, server] = rpc::loopbackPair();
    ScriptedWorker worker(std::move(server));
    RemoteWorker rw(std::move(client));
    test::TempFile bundle("foundation", ".ltb");
    { std::ofstream(bundle.path) << "stub"; }

    const Dims5 dims{1, 2, 4, 8, 8};
    auto array = std::make_shared<Array5>(Array5::zeros(dims));
    DatasetMeta meta;
    meta.dims = dims;
    meta.voxelUm = {0.15, 0.15, 0.75};
    meta.normalizeChannels();
    const Operation& op = requireOperation("foundation");
    ParamSet p = op.defaults();
    p.set("model", bundle.str);
    StepContext ctx;
    ctx.remote = &rw;
    const auto sent = [&worker] {
        std::lock_guard<std::mutex> lock(worker.sentMutex);
        return worker.foundationParams;
    };

    SECTION("Detect with a Min. voxels keeps its one-voxel objects") {
        // the step used to run its own size filter after the worker's, on
        // every task but Track: a detection is one voxel, so all were removed
        p.set("task", std::string("Detect centroids"));
        p.set("min_voxels", std::int64_t{5});
        const StepOutput out = op.run(StepInput{meta, array, nullptr, nullptr}, p, ctx);
        REQUIRE(out.labels);
        CHECK(out.labels->at(0, 1, 1, 1) == 1u);
        CHECK(out.labels->at(1, 3, 7, 7) == 2u);
        CHECK(out.labels->stats().size() >= 2);
        // the note carries the summary: it was read after the diagnostics were moved from
        CHECK_THAT(out.note, Catch::Matchers::ContainsSubstring("detect · 2 labels"));
        CHECK_THAT(out.diagnostics.summary, Catch::Matchers::ContainsSubstring("2 labels"));
        const json params = sent();
        CHECK(params.value("task", "") == "detect");
        // (x, y, z), the application's order; the worker turns it into latents' (z, y, x)
        CHECK(params["voxel_um"] == json::array({0.15, 0.15, 0.75}));
        CHECK_FALSE(params.contains("tile"));   // all zero: the bundle's own
    }
    SECTION("Segment keeps small objects as the worker sent them") {
        p.set("min_voxels", std::int64_t{5});
        const StepOutput out = op.run(StepInput{meta, array, nullptr, nullptr}, p, ctx);
        REQUIRE(out.labels);
        CHECK(out.labels->at(1, 1, 1, 1) == 1u);
        CHECK(sent().value("min_voxels", 0) == 5);   // the worker applies it
    }
    SECTION("a tracking run is tracked and its division count says it is approximate") {
        p.set("task", std::string("Track over time"));
        const StepOutput out = op.run(StepInput{meta, array, nullptr, nullptr}, p, ctx);
        REQUIRE(out.labels);
        CHECK(out.labels->tracked());
        CHECK_THAT(out.note, Catch::Matchers::ContainsSubstring("2 tracks"));
        CHECK(std::any_of(out.diagnostics.facts.begin(), out.diagnostics.facts.end(),
                          [](const DiagnosticFact& f) { return f.key == "Divisions (approx.)" && f.value == "1"; }));
    }
    SECTION("a tile with some extents given is sent, zero meaning the bundle's on that axis") {
        p.set("tile", std::vector<double>{0, 32, 32});
        (void)op.run(StepInput{meta, array, nullptr, nullptr}, p, ctx);
        CHECK(sent()["tile"] == json::array({0, 32, 32}));
    }
    SECTION("a bundle this machine cannot see is left to the worker") {
        // A bundle picked from a registry on a cluster lives on the worker's
        // filesystem. Refusing it here, because it is not on this machine,
        // made the HPC backend unable to run any registry bundle.
        const std::string remote = "/cluster/only/registry/track-nih-ls-2d-pretrained.ltb";
        REQUIRE_FALSE(std::filesystem::exists(remote));
        p.set("model", remote);
        const Validation v = op.validate(p, meta);
        CHECK(v.ok());
        REQUIRE(v.warnings.size() >= 1);
        CHECK_THAT(v.warnings.front(), Catch::Matchers::ContainsSubstring("not found on this machine"));
        const StepOutput out = op.run(StepInput{meta, array, nullptr, nullptr}, p, ctx);
        REQUIRE(out.labels);
        CHECK(sent().value("model", "") == remote);   // sent as given
        p.set("model", std::string());
        CHECK_FALSE(op.validate(p, meta).ok());        // no model at all is still an error
    }
    SECTION("Min. voxels is shown for Segment only") {
        const auto spec = std::find_if(op.info().params.begin(), op.info().params.end(),
                                       [](const ParamSpec& s) { return s.key == "min_voxels"; });
        REQUIRE(spec != op.info().params.end());
        CHECK(spec->visibleFor(p));
        p.set("task", std::string("Detect centroids"));
        CHECK_FALSE(spec->visibleFor(p));
        p.set("task", std::string("Track over time"));
        CHECK_FALSE(spec->visibleFor(p));
    }
}

TEST_CASE("connectTcp reports an unreachable port", "[app][rpc]") {
    CHECK_THROWS(rpc::connectTcp("127.0.0.1", 1, std::chrono::milliseconds(500)));
    (void)workerScriptPath("/definitely/not/here");   // must not throw
}

TEST_CASE("workerScriptPath finds an installed worker before the build tree's and the checkout's", "[app][rpc]") {
    namespace fs = std::filesystem;
    const fs::path prefix = fs::temp_directory_path() / "sirius-installed-worker-test";
    fs::remove_all(prefix);
    const fs::path bin = prefix / "bin";
    auto plant = [](const fs::path& dir) {
        fs::create_directories(dir / "sirius_worker");
        std::ofstream(dir / "sirius_worker" / "__main__.py") << "\n";
        return dir.lexically_normal();
    };
    const fs::path installed = plant(bin / installedDataDirectoryFromBindir() / "python");
    const fs::path beside = plant(bin / "python");
    struct Restore {
        ~Restore() { setApplicationDirectory({}); }
    } restore;
    setApplicationDirectory(bin.string());
    CHECK(fs::path(workerScriptPath()) == installed);
    // a directory the caller names (Preferences) wins over both
    const fs::path chosen = plant(prefix / "chosen");
    CHECK(fs::path(workerScriptPath(chosen.string())) == chosen);
    // not installed: the copy the build puts beside the executable
    fs::remove_all(prefix / "share");
    CHECK(fs::path(workerScriptPath()) == beside);
    fs::remove_all(prefix);
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

TEST_CASE("rpc tensor descriptors must hold non-negative integers", "[app][rpc]") {
    // 1e309 parses as an infinite double; get<Index>() of that is undefined
    // (the Python worker died of int(inf)). Every such descriptor is a
    // protocol error, whatever the JSON parser made of the number.
    auto put32 = [](std::vector<std::byte>& out, std::uint32_t v) {   // little-endian, as the wire is
        for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
    };
    auto put64 = [](std::vector<std::byte>& out, std::uint64_t v) {
        for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
    };
    auto frame = [&](const std::string& header, std::uint64_t payloadLen) {
        std::vector<std::byte> out;
        put32(out, static_cast<std::uint32_t>(header.size()));
        for (char c : header) out.push_back(static_cast<std::byte>(c));
        put64(out, payloadLen);
        out.resize(out.size() + payloadLen, std::byte{0});
        return out;
    };
    auto descriptor = [](const std::string& shape, const std::string& offset, const std::string& nbytes) {
        return "{\"id\":1,\"type\":\"result\",\"tensors\":[{\"name\":\"a\",\"dtype\":\"float32\",\"shape\":" + shape +
               ",\"offset\":" + offset + ",\"nbytes\":" + nbytes + "}]}";
    };
    for (const char* shape : {"[1e309]", "[1.5]", "[-1]", "[\"2\"]", "[true]", "2"}) {
        INFO("shape " << shape);
        std::vector<std::byte> bytes = frame(descriptor(shape, "0", "4"), 4);
        CHECK_THROWS_AS(rpc::decodeFrame(bytes), ProtocolError);
    }
    for (const char* offset : {"1e309", "0.5", "-1", "\"0\""}) {
        INFO("offset " << offset);
        std::vector<std::byte> bytes = frame(descriptor("[1]", offset, "4"), 4);
        CHECK_THROWS_AS(rpc::decodeFrame(bytes), ProtocolError);
    }
    for (const char* nbytes : {"1e309", "4.0", "-4"}) {
        INFO("nbytes " << nbytes);
        std::vector<std::byte> bytes = frame(descriptor("[1]", "0", nbytes), 4);
        CHECK_THROWS_AS(rpc::decodeFrame(bytes), ProtocolError);
    }
    std::vector<std::byte> good = frame(descriptor("[1]", "0", "4"), 4);
    const auto m = rpc::decodeFrame(good);
    REQUIRE(m);
    REQUIRE(m->tensors.size() == 1);
    CHECK(m->tensors[0].numel() == 1);
    // and the sender checks its own arithmetic the same way
    std::vector<float> one{1.f};
    std::vector<rpc::TensorRef> wrapped{{"a", "float32", {std::numeric_limits<Index>::max(), 4}, one.data(), 4}};
    CHECK_THROWS(rpc::encodeFrame({{"id", 1}}, wrapped));
}

// --- plugin reloads --------------------------------------------------------------

#include <filesystem>
#include <fstream>
#include <mutex>

#include "core/array_source.hpp"
#include "core/help_pages.hpp"
#include "core/ops/plugin.hpp"
#include "core/pipeline.hpp"
#include "core/workbench.hpp"

#include "temp_path.hpp"

namespace {

    // A worker whose plugin list the test sets: it answers hello and
    // list_plugins / reload_plugins, one loopback connection per launch.
    struct PluginCatalog {
        struct Connection {
            std::unique_ptr<rpc::Transport> t;
            std::thread thread;
            Connection(std::unique_ptr<rpc::Transport> transport, PluginCatalog& catalog) : t(std::move(transport)) {
                thread = std::thread([this, &catalog] { serve(catalog); });
            }
            ~Connection() {
                t->close();
                thread.join();
            }
            void serve(PluginCatalog& catalog) {
                std::vector<std::byte> buf;
                try {
                    for (;;) {
                        auto m = rpc::decodeFrame(buf);
                        if (!m) {
                            t->receive(buf, std::chrono::milliseconds(50));
                            continue;
                        }
                        const std::uint64_t id = m->header.value("id", 0ull);
                        const std::string method = m->header.value("method", "");
                        json result;
                        if (method == "hello")
                            result = {{"version", "test"}, {"methods", json::array()}, {"protocol_version", rpc::kProtocolVersion}, {"device", "cpu"}, {"hostname", "loop"}};
                        else if (method == "list_plugins" || method == "reload_plugins")
                            result = {{"plugins", catalog.plugins()}, {"dirs", {"/plugins"}}};
                        if (result.is_null()) t->send(rpc::encodeFrame({{"id", id}, {"type", "error"}, {"message", "unknown method " + method}}, {}));
                        else t->send(rpc::encodeFrame({{"id", id}, {"type", "result"}, {"result", result}}, {}));
                    }
                } catch (const std::exception&) {
                    // the client closed: done
                }
            }
        };

        std::mutex mutex;
        json list = json::array();
        std::vector<std::unique_ptr<Connection>> connections;

        json plugins() {
            std::lock_guard<std::mutex> g(mutex);
            return list;
        }
        void set(json l) {
            std::lock_guard<std::mutex> g(mutex);
            list = std::move(l);
        }
        std::unique_ptr<RemoteWorker> connect() {
            auto [client, server] = rpc::loopbackPair();
            connections.push_back(std::make_unique<Connection>(std::move(server), *this));
            return std::make_unique<RemoteWorker>(std::move(client));
        }
    };

    json pluginSpec(const std::string& kind, double offset) {
        return {{"kind", kind},
                {"name", kind},
                {"file", "/plugins/" + kind + ".py"},
                {"params", json::array({{{"key", "gain"}, {"type", "double"}, {"default", 1.0}},
                                        {{"key", "offset"}, {"type", "double"}, {"default", offset}}})},
                {"help", "# " + kind + "\n\nA test plugin.\n"}};
    }

    bool inAddMenu(const std::string& kind) {
        for (const auto& group : operationGroups())
            for (const Operation* op : group.second)
                if (op->kind() == kind) return true;
        return false;
    }

    bool inPluginKinds(const std::string& kind) {
        const std::vector<std::string> kinds = pluginKinds();
        return std::find(kinds.begin(), kinds.end(), kind) != kinds.end();
    }

} // namespace

TEST_CASE("A plugin whose file is gone stays in the pipeline as not loaded", "[app][rpc][plugin]") {
    registerBuiltinOperations();
    const std::filesystem::path scratch = test::uniqueTempPath("plugin_catalog", "");
    PluginCatalog catalog;
    {
        Workbench wb(scratch);
        wb.setLocalWorkerLauncher([&catalog] { return catalog.connect(); });
        auto array = std::make_shared<Array5>(Array5::filled(Dims5{1, 1, 2, 4, 4}, 1.0f));
        DatasetMeta meta;
        meta.name = "synthetic";
        meta.sourcePath = "memory://synthetic";
        meta.dims = array->dims();
        wb.setDataset(std::make_shared<MemorySource>(array, meta));

        // a pipeline opened before its plugin is loaded
        wb.replacePipeline(Pipeline::fromJson({{"steps", json::array({{{"kind", "load"}}, {{"kind", "zz_late"}, {"params", {{"gain", 3.0}}}}})}}),
                           "Load pipeline");
        REQUIRE(wb.pipeline().at(1).op().info().missing);
        catalog.set(json::array({pluginSpec("zz_late", 5.0), pluginSpec("zz_gone", 0.0), pluginSpec("zz_broken", 0.0)}));
        CHECK(wb.loadPlugins(false) == 3);
        CHECK_FALSE(wb.pipeline().at(1).op().info().missing);
        CHECK(wb.pipeline().at(1).params.getDouble("gain") == 3.0);     // as the pipeline said
        CHECK(wb.pipeline().at(1).params.getDouble("offset") == 5.0);   // declared by the plugin
        CHECK(inAddMenu("zz_gone"));
        CHECK(inPluginKinds("zz_gone"));
        wb.addStep("zz_gone");
        wb.setStepParam(2, "gain", 2.0);

        // zz_gone.py is deleted; zz_broken.py now fails to import (and is listed by its file name)
        catalog.set(json::array({pluginSpec("zz_late", 5.0),
                                 {{"kind", "zz_broken"}, {"name", "zz_broken"}, {"file", "/plugins/zz_broken.py"}, {"error", "SyntaxError: invalid syntax"}}}));
        CHECK(wb.loadPlugins(true) == 1);
        const Operation* gone = findOperation("zz_gone");
        REQUIRE(gone);
        CHECK(gone->info().missing);
        CHECK_FALSE(inAddMenu("zz_gone"));       // was still offered by the add menu
        CHECK_FALSE(inPluginKinds("zz_gone"));
        CHECK(loadHelpPage("zz_gone").intro.find("A test plugin") == std::string::npos);
        const Operation* broken = findOperation("zz_broken");
        REQUIRE(broken);
        CHECK_FALSE(broken->info().missing);   // a file still there keeps its last good registration
        CHECK(inAddMenu("zz_broken"));
        // the step keeps its place, its parameters and a reason
        REQUIRE(wb.pipeline().size() == 3);
        CHECK(wb.pipeline().at(2).kind == "zz_gone");
        CHECK(wb.pipeline().at(2).params.getDouble("gain") == 2.0);
        CHECK_THAT(wb.stepValidation(2).firstError(), Catch::Matchers::ContainsSubstring("not loaded"));
        CHECK_THROWS_AS(wb.addStep("zz_gone"), std::out_of_range);
        wb.undo();
        wb.undo();
        CHECK(wb.pipeline().size() == 2);
        wb.redo();   // a snapshot naming the kind restores
        REQUIRE(wb.pipeline().size() == 3);
        CHECK(wb.pipeline().at(2).kind == "zz_gone");

        // the file comes back
        catalog.set(json::array({pluginSpec("zz_late", 5.0), pluginSpec("zz_gone", 0.0), pluginSpec("zz_broken", 0.0)}));
        CHECK(wb.loadPlugins(true) == 3);
        CHECK_FALSE(findOperation("zz_gone")->info().missing);
        CHECK(inAddMenu("zz_gone"));
        CHECK(wb.stepValidation(2).ok());
    }
    std::error_code ec;
    std::filesystem::remove_all(scratch, ec);
}

#ifndef _WIN32
TEST_CASE("a plugin file deleted and reloaded leaves the add menu", "[app][rpc][worker][plugin]") {
    const char* python = std::getenv("SIRIUS_PYTHON");
    if (!python || !*python) SKIP("SIRIUS_PYTHON is not set");
    const std::string dir = workerScriptPath();
    if (dir.empty()) SKIP("sirius_worker not found");
    registerBuiltinOperations();
    const std::filesystem::path pdir = test::uniqueTempPath("plugins_reload", "");
    std::filesystem::create_directories(pdir);
    for (const char* kind : {"zz_worker_gone", "zz_worker_kept"})
        std::ofstream(pdir / (std::string(kind) + ".py")) << "STEP = {'kind': '" << kind << "', 'name': 'Test'}\n"
                                                          << "def run(data, params, meta, ctx):\n    return data\n";
    const std::string cmd = std::string("cd '") + dir + "' && SIRIUS_PLUGIN_DIRS='" + pdir.string() + "' exec '" + python +
                            "' -m sirius_worker --host 127.0.0.1 --port 0 --token reload --device cpu 2>/dev/null";
    FILE* pipe = ::popen(cmd.c_str(), "r");
    REQUIRE(pipe);
    char line[512] = {0};
    REQUIRE(std::fgets(line, sizeof line, pipe));
    const int port = json::parse(line).value("port", 0);
    REQUIRE(port > 0);
    auto worker = RemoteWorker::connect("127.0.0.1", port, "reload");

    PluginLoadResult r = registerPluginOperations(*worker, false);
    CHECK(inAddMenu("zz_worker_gone"));
    CHECK(inAddMenu("zz_worker_kept"));
    Pipeline p;
    p.add("zz_worker_gone");
    const json snapshot = p.toJson();

    std::filesystem::remove(pdir / "zz_worker_gone.py");   // Plugin Manager ▸ Delete
    r = registerPluginOperations(*worker, true);
    CHECK(r.removed == std::vector<std::string>{"zz_worker_gone"});
    REQUIRE(findOperation("zz_worker_gone"));
    CHECK(findOperation("zz_worker_gone")->info().missing);
    CHECK_FALSE(inAddMenu("zz_worker_gone"));
    CHECK_FALSE(inPluginKinds("zz_worker_gone"));
    CHECK(inAddMenu("zz_worker_kept"));
    CHECK_NOTHROW(Pipeline::fromJson(snapshot));

    // a file that breaks is not a removal
    std::ofstream(pdir / "zz_worker_kept.py") << "this is not python (\n";
    r = registerPluginOperations(*worker, true);
    CHECK(r.removed.empty());
    CHECK_FALSE(findOperation("zz_worker_kept")->info().missing);
    CHECK(inAddMenu("zz_worker_kept"));

    (void)worker->call("shutdown", json::object());
    worker->close();
    ::pclose(pipe);
    std::filesystem::remove_all(pdir);
}
#endif

// --- a worker that dies ------------------------------------------------------------

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

TEST_CASE("A worker that dies while a request is sent is an error, not SIGPIPE", "[app][rpc]") {
    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(listener >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    REQUIRE(::bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
    REQUIRE(::listen(listener, 1) == 0);
    socklen_t len = sizeof addr;
    REQUIRE(::getsockname(listener, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
    // answers hello, then its process is gone (out of memory, a wall-time limit)
    std::thread server([listener] {
        const int s = ::accept(listener, nullptr, nullptr);
        if (s < 0) return;
        std::vector<std::byte> in;
        std::vector<char> buf(1 << 16);
        std::optional<rpc::Message> hello;
        while (!(hello = rpc::decodeFrame(in))) {
            const auto n = ::recv(s, buf.data(), buf.size(), 0);
            if (n <= 0) break;
            in.insert(in.end(), reinterpret_cast<const std::byte*>(buf.data()), reinterpret_cast<const std::byte*>(buf.data()) + n);
        }
        if (hello) {
            const std::vector<std::byte> reply = rpc::encodeFrame(
                {{"id", hello->header["id"]}, {"type", "result"}, {"result", {{"protocol_version", rpc::kProtocolVersion}}}}, {});
            (void)::send(s, reply.data(), reply.size(), 0);
        }
        ::close(s);
    });
    std::unique_ptr<RemoteWorker> worker = RemoteWorker::connect("127.0.0.1", ntohs(addr.sin_port), "");
    server.join();
    ::close(listener);
    std::vector<float> volume(16 << 20);   // 64 MB: more than the socket buffers take without a reader
    const rpc::TensorRef ref{"volume", "float32", {static_cast<Index>(volume.size())}, volume.data(), volume.size() * sizeof(float)};
    // before: the application ended here with SIGPIPE (exit status 141)
    CHECK_THROWS_AS(worker->call("run", {{"kind", "x"}}, {ref}), ProtocolError);
}
#endif
