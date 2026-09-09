// Numeric parity fixtures for the Python mirror of the operations.
//
// bindings/python/sirius/workbench.py reimplements a dozen of the built-in
// steps for the exported scripts and the HPC backend. test_app_schema.cpp
// checks their *parameter keys* against a committed snapshot; nothing checked
// that the two implementations produce the same *numbers*, which is the
// failure that would corrupt someone's results without saying anything.
//
// This case runs a fixed list of (kind, params) through the real Operation on
// one deterministic synthetic array and writes every result as raw float32
// with a JSON sidecar. bindings/tests/test_parity.py replays the same list
// through sirius.workbench.run_step on the same input and compares.
//
//     SIRIUS_PARITY_OUT=<dir> sirius_tests "[parity]"
//     SIRIUS_PARITY_DIR=<dir> python -m unittest bindings.tests.test_parity
//
// The tag is hidden ("[.parity]"), so the normal test run neither runs it nor
// pays for it; without SIRIUS_PARITY_OUT it skips.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/operation.hpp"
#include "core/ops/builtin.hpp"

using namespace sirius;
using namespace sirius::app;
using nlohmann::json;

namespace {

    // (c, t, z, y, x) chosen small enough to compare voxel by voxel and odd
    // enough that a transposed axis or an off-by-one stride cannot hide.
    constexpr Index kC = 2, kT = 3, kZ = 4, kY = 9, kX = 11;

    // Three Gaussian blobs on a deterministic hash noise floor: smooth enough
    // that resampling means something, lumpy enough that a threshold finds
    // several connected components to label.
    std::shared_ptr<Array5> syntheticArray(Dims5 d) {
        auto a = std::make_shared<Array5>(d);
        const double blobs[3][4] = {{1.0, 2.5, 3.0, 1.0}, {2.5, 6.0, 7.5, 0.8}, {0.5, 7.0, 2.0, 0.6}};
        for (Index c = 0; c < d.c; ++c)
            for (Index t = 0; t < d.t; ++t)
                for (Index z = 0; z < d.z; ++z)
                    for (Index y = 0; y < d.y; ++y)
                        for (Index x = 0; x < d.x; ++x) {
                            double v = 0.0;
                            for (const auto& b : blobs) {
                                const double dz = z - b[0], dy = y - b[1], dx = x - b[2];
                                v += b[3] * std::exp(-(dz * dz / 2.0 + dy * dy / 6.0 + dx * dx / 6.0));
                            }
                            v *= 1.0 + 0.15 * c - 0.05 * t;   // channels and time points differ
                            // integer hash noise: identical on every platform
                            std::uint32_t h = static_cast<std::uint32_t>(((c * 31 + t) * 31 + z) * 31 + y) * 31u +
                                              static_cast<std::uint32_t>(x);
                            h ^= h >> 15;
                            h *= 0x2c1b3c6du;
                            h ^= h >> 12;
                            v += 0.05 * (static_cast<double>(h % 1000u) / 1000.0);
                            a->at(c, t, z, y, x) = static_cast<float>(v);
                        }
        return a;
    }

    DatasetMeta syntheticMeta(Dims5 d) {
        DatasetMeta m;
        m.name = "parity";
        m.format = "memory";
        m.dims = d;
        m.voxelUm = {0.1, 0.1, 0.3};   // x, y, z
        m.normalizeChannels();
        return m;
    }

    void writeFloats(const std::filesystem::path& p, const float* data, std::size_t n) {
        std::ofstream f(p, std::ios::binary);
        REQUIRE(f.good());
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
        REQUIRE(f.good());
    }

    void writeJson(const std::filesystem::path& p, const json& j) {
        std::ofstream f(p);
        REQUIRE(f.good());
        f << j.dump(2) << '\n';
        REQUIRE(f.good());
    }

    json dimsJson(const Dims5& d) { return json::array({d.c, d.t, d.z, d.y, d.x}); }

    struct Case {
        const char* name;
        const char* kind;
        json params;
    };

    // One case per behaviour the two implementations are meant to share.
    // Steps left out on purpose: merge (its output is a display RGB blend
    // whose channel colours come from the metadata, not from the array),
    // flatfield and load (both read files), and every worker-backed kind.
    const std::vector<Case> kCases = {
        {"einsum_mean_t", "einsum", {{"keep", "czyx"}, {"reduction", "mean"}}},
        {"einsum_sum_zyx", "einsum", {{"keep", "ct"}, {"reduction", "sum"}}},
        {"einsum_max_c", "einsum", {{"keep", "tzyx"}, {"reduction", "max"}}},
        {"einsum_min_yx", "einsum", {{"keep", "ctz"}, {"reduction", "min"}}},
        {"maxproj_z", "maxproj", {{"axis", "z"}}},
        {"maxproj_t", "maxproj", {{"axis", "t"}}},
        {"meant", "meant", json::object()},
        {"contrast_auto", "contrast", {{"min", 0.0}, {"max", 0.0}, {"gamma", 1.0}}},
        {"contrast_percentiles", "contrast", {{"min", 0.0}, {"max", 0.0}, {"lo_percentile", 5.0}, {"hi_percentile", 95.0}}},
        {"contrast_manual", "contrast", {{"min", 0.25}, {"max", 0.8}, {"gamma", 1.0}}},
        {"contrast_gamma", "contrast", {{"min", 0.1}, {"max", 0.9}, {"gamma", 0.45}}},
        {"croppad_crop", "croppad", {{"z0", 1}, {"y0", 2}, {"x0", 3}, {"z", 2}, {"y", 4}, {"x", 5}}},
        {"croppad_pad", "croppad", {{"z0", -1}, {"y0", -2}, {"x0", -2}, {"z", 6}, {"y", 12}, {"x", 14}, {"fill", 0.25}}},
        {"croppad_to_edge", "croppad", {{"z0", 1}, {"y0", 1}, {"x0", 1}}},
        {"bleach_first_t", "bleach", {{"mode", "Match first frame"}, {"over", "t"}}},
        {"bleach_mean_t", "bleach", {{"mode", "Match mean"}, {"over", "t"}}},
        {"bleach_mean_z", "bleach", {{"mode", "Match mean"}, {"over", "z"}}},
        {"resample_linear", "resample", {{"voxel_x", 0.16}, {"voxel_y", 0.16}, {"voxel_z", 0.2}, {"interpolation", "linear"}}},
        {"resample_up_linear", "resample", {{"voxel_x", 0.06}, {"voxel_y", 0.08}, {"voxel_z", 0.0}, {"interpolation", "linear"}}},
        {"resample_cubic", "resample", {{"voxel_x", 0.16}, {"voxel_y", 0.16}, {"voxel_z", 0.2}, {"interpolation", "cubic"}}},
        {"resample_nearest", "resample", {{"voxel_x", 0.16}, {"voxel_y", 0.16}, {"voxel_z", 0.2}, {"interpolation", "nearest"}}},
        {"threshold_otsu", "threshold", {{"channel", 0}, {"method", "Otsu"}, {"post", "Connected components"}, {"min_voxels", 0}}},
        {"threshold_manual", "threshold", {{"channel", 1}, {"method", "Manual"}, {"value", 0.6}, {"post", "Connected components"}, {"min_voxels", 4}}},
        {"threshold_percentile", "threshold", {{"channel", 0}, {"method", "Percentile"}, {"percentile", 92.0}, {"post", "Connected components"}, {"min_voxels", 0}}},
        // classical segmentation: one case per branch that has its own maths,
        // so the Python mirror cannot drift from the C++ on any of them
        {"classic_otsu_hmax", "classic", {{"channel", 0}, {"method", "Otsu"}, {"sigma", 1.0}, {"opening", 1}, {"post", "Watershed (distance)"}, {"seeds", "H-maxima"}, {"seed_depth", 1.5}, {"min_voxels", 4}}},
        // No "Distance maxima" case: with those seeds this fixture puts two
        // seeds equidistant from the ridge between them, and the two floods
        // break that tie differently -- the application's priority queue and
        // scikit-image's give 22 voxels of one shared boundary to different
        // neighbours. The foreground and the object count agree; only the
        // border moves. Matching it would mean reimplementing the C++ queue
        // order in the mirror. The h-maxima case below covers the same
        // watershed code with seeds that are not tied.
        {"classic_multi_otsu", "classic", {{"channel", 1}, {"method", "Multi-Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_local_contrast", "classic", {{"channel", 0}, {"method", "Local contrast"}, {"window", 11}, {"contrast_k", 1.2}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_local_mean", "classic", {{"channel", 0}, {"method", "Local mean"}, {"window", 11}, {"local_ratio", 1.15}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_blobs", "classic", {{"channel", 0}, {"enhance", "Blobs (DoG)"}, {"enhance_sigma", 1.5}, {"method", "Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        // No "Blob centres (LoG)" case either. Inside a smooth blob the
        // scale-normalised response is a broad, nearly flat cap, so which
        // voxel of it wins the peak comes down to the last bits of a Gaussian
        // -- and the two Gaussians (this one and scipy's) do not agree there.
        // A seed one voxel over moves the boundary the flood then draws. The
        // property that matters, one seed per object whatever its size, is
        // asserted directly in tests/test_app_tracking.cpp.
        {"classic_tubes", "classic", {{"channel", 0}, {"enhance", "Tubes (Frangi)"}, {"enhance_sigma", 1.0}, {"enhance_sigma_max", 3.0}, {"enhance_scales", 3}, {"method", "Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_tophat", "classic", {{"channel", 0}, {"tophat", 4}, {"method", "Otsu"}, {"sigma", 1.0}, {"opening", 1}, {"post", "Connected components"}, {"min_voxels", 4}}},
        {"classic_triangle", "classic", {{"channel", 0}, {"method", "Triangle"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_li", "classic", {{"channel", 0}, {"method", "Li"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_median", "classic", {{"channel", 0}, {"denoise", "Median 3x3"}, {"method", "Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_hysteresis", "classic", {{"channel", 0}, {"method", "Otsu"}, {"hysteresis", true}, {"hysteresis_ratio", 0.6}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_snake", "classic", {{"channel", 0}, {"method", "Otsu"}, {"refine", "Active contour (Chan-Vese)"}, {"refine_iterations", 6}, {"refine_smoothing", 1}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_gradient_watershed", "classic", {{"channel", 0}, {"method", "Otsu"}, {"sigma", 1.0}, {"opening", 1}, {"post", "Watershed (gradient)"}, {"seeds", "H-maxima"}, {"seed_depth", 1.5}, {"min_voxels", 4}}},
        // two objects before the filter, one after: the fixture would pass on a
        // filter that did nothing if both survived it
        {"classic_shape_filter", "classic", {{"channel", 0}, {"method", "Multi-Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 1}, {"max_voxels", 30}, {"max_elongation", 8.0}}},
        {"classic_shape_fill", "classic", {{"channel", 0}, {"method", "Multi-Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 1}, {"min_fill", 0.5}}},
        {"classic_yen", "classic", {{"channel", 0}, {"method", "Yen"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_isodata", "classic", {{"channel", 0}, {"method", "Isodata"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_neurites", "classic", {{"channel", 0}, {"enhance", "Neurites (Meijering)"}, {"enhance_sigma", 1.0}, {"enhance_sigma_max", 3.0}, {"enhance_scales", 3}, {"method", "Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        // This fixture's 4 x 9 x 11 volume encloses no cavity, so what the
        // case is worth is that both sides agree the 3D fill changes nothing.
        // The fill itself is asserted on volumes built for it in
        // tests/test_app_labels.cpp.
        {"classic_holes_3d", "classic", {{"channel", 0}, {"method", "Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"fill_holes_3d", true}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_expand", "classic", {{"channel", 0}, {"method", "Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}, {"expand", 2.0}}},
        {"classic_rolling_ball", "classic", {{"channel", 0}, {"background", "Rolling ball"}, {"tophat", 4}, {"method", "Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        // radius 25: the ball runs on a 3x decimated copy (lround(2.5) = 3,
        // where Python's round would say 2) -- the radius the presets use
        {"classic_rolling_ball_25", "classic", {{"channel", 0}, {"background", "Rolling ball"}, {"tophat", 25}, {"method", "Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}}},
        {"classic_skeleton", "classic", {{"channel", 0}, {"method", "Otsu"}, {"sigma", 0.0}, {"opening", 0}, {"fill_holes", false}, {"post", "Connected components"}, {"min_voxels", 2}, {"skeleton", true}}},
        // No "Anisotropic diffusion" case: every step of it evaluates exp(),
        // and the C++ standard library and NumPy do not agree in the last bit.
        // Five iterations later a voxel can land on the other side of the
        // threshold. The mirror follows the same formula and is asserted on
        // its own behaviour (an edge kept, an interior flattened) in
        // tests/test_app_labels.cpp.
    };

    ParamSet paramsOf(const json& j) {
        ParamSet p;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it->is_boolean()) p.set(it.key(), it->get<bool>());
            else if (it->is_number_integer()) p.set(it.key(), it->get<std::int64_t>());
            else if (it->is_number_float()) p.set(it.key(), it->get<double>());
            else p.set(it.key(), it->get<std::string>());
        }
        return p;
    }

} // namespace

TEST_CASE("parity fixtures for the Python mirror of the operations", "[.parity][app]") {
    const char* outDir = std::getenv("SIRIUS_PARITY_OUT");
    if (!outDir || !*outDir) SKIP("set SIRIUS_PARITY_OUT to a directory to write the parity fixtures");
    const std::filesystem::path dir(outDir);
    std::filesystem::create_directories(dir);

    registerBuiltinOperations();
    const Dims5 dims{kC, kT, kZ, kY, kX};
    const DatasetMeta meta = syntheticMeta(dims);
    const std::shared_ptr<Array5> array = syntheticArray(dims);

    writeFloats(dir / "input.f32", array->data(), static_cast<std::size_t>(array->numel()));
    writeJson(dir / "input.json", json{{"dims", dimsJson(dims)},
                                       {"voxel_um", json::array({meta.voxelUm[0], meta.voxelUm[1], meta.voxelUm[2]})}});

    StepContext ctx;
    ctx.backend = Backend::Cpu;
    json index = json::array();
    for (const Case& c : kCases) {
        INFO("case " << c.name);
        const Operation* op = findOperation(c.kind);
        REQUIRE(op != nullptr);
        // The executor hands every step a fully defaulted parameter set; the
        // sidecar records the same effective values so the Python side runs
        // the same step and not its own idea of the defaults.
        ParamSet params = paramsOf(c.params);
        params.applyDefaults(op->info().params);
        StepInput in;
        in.meta = meta;
        in.array = array;
        const StepOutput out = op->run(in, params, ctx);
        REQUIRE(out.array);

        json entry{{"name", c.name}, {"kind", c.kind}, {"params", params.toJson()}, {"dims", dimsJson(out.meta.dims)}, {"voxel_um", json::array({out.meta.voxelUm[0], out.meta.voxelUm[1], out.meta.voxelUm[2]})}, {"labels", false}};
        writeFloats(dir / (std::string(c.name) + ".f32"), out.array->data(),
                    static_cast<std::size_t>(out.array->numel()));
        if (out.labels && !out.labels->empty()) {
            entry["labels"] = true;
            entry["labels_dims"] = json::array({out.labels->t(), out.labels->z(), out.labels->y(), out.labels->x()});
            std::vector<std::uint32_t> flat;
            flat.reserve(static_cast<std::size_t>(out.labels->t() * out.labels->z() * out.labels->y() * out.labels->x()));
            for (Index t = 0; t < out.labels->t(); ++t) {
                const std::uint32_t* v = out.labels->volume(t);
                flat.insert(flat.end(), v, v + out.labels->z() * out.labels->y() * out.labels->x());
            }
            std::ofstream f(dir / (std::string(c.name) + ".u32"), std::ios::binary);
            REQUIRE(f.good());
            f.write(reinterpret_cast<const char*>(flat.data()),
                    static_cast<std::streamsize>(flat.size() * sizeof(std::uint32_t)));
            REQUIRE(f.good());
        }
        writeJson(dir / (std::string(c.name) + ".json"), entry);
        index.push_back(entry);
    }
    writeJson(dir / "cases.json", json{{"version", 1}, {"cases", index}});
}
