// The app-side OTF measurement service (app/core/otf_measure.hpp): what a
// dataset can fill in by itself, the ONE validate the dialog, the tool API and
// the bindings share, the frame gather that reads a raw stack through its own
// layout instead of assuming an order, and what a run leaves behind -- the
// table in makeotf's form, its sidecar, the provenance that ties it to the
// acquisition, and the diagnostics the dock's cells draw.
//
// The physics is the library's (tests/test_otf_measure.cpp, which checks the
// table against closed-form answers and against the colleague's makeotf file).
// Nothing here re-checks it: these cases are about the layer between a dataset
// and that measurement, and the one claim they do make about the numbers is
// that two storage orders of the SAME acquisition must give the SAME table.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <sirius/tiff_io.hpp>

#include "core/otf_measure.hpp"
#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;

namespace {

    constexpr double kDxy = 0.0855, kDz = 0.1;   // the iSOAR2 instrument's own, in micrometres
    // not the POSIX macro, which is not standard C++ and which nothing else in this tree uses
    constexpr double kTwoPi = 6.283185307179586;

    // One or more Gaussian beads under a lateral sinusoid, as a function of
    // the LOGICAL frame (angle, phase, z). Where those frames are stored is
    // the layout's business, which is the point of the cases below.
    struct Scene {
        Index nz = 12, ny = 64, nx = 64;
        int nphases = 3, nangles = 1;
        double background = 100.0, modulation = 0.7;
        double sigmaXY = 1.6, sigmaZ = 3.0, periodPx = 8.0;
        std::vector<std::array<double, 4>> beads{{32.0, 32.0, 6.0, 3000.0}};   // x, y, z, amplitude
        double angleGain = 1.0;   // amplitude of direction d is gain^d

        double value(Index angle, Index phase, Index z, Index y, Index x) const {
            const double phi = kTwoPi * static_cast<double>(phase) / static_cast<double>(nphases);
            const double illum = 1.0 + modulation * std::cos(kTwoPi * static_cast<double>(x) / periodPx + phi);
            double v = background;
            for (const std::array<double, 4>& b : beads) {
                const double dx = static_cast<double>(x) - b[0], dy = static_cast<double>(y) - b[1],
                             dz = static_cast<double>(z) - b[2];
                const double g = std::exp(-(dx * dx + dy * dy) / (2.0 * sigmaXY * sigmaXY) -
                                          dz * dz / (2.0 * sigmaZ * sigmaZ));
                v += b[3] * std::pow(angleGain, static_cast<double>(angle)) * g * illum;
            }
            return v;
        }
    };

    Dims5 dimsOf(const Scene& s) {
        Dims5 d;
        d.c = 1;
        d.t = 1;
        d.z = static_cast<Index>(s.nangles) * s.nphases * s.nz;
        d.y = s.ny;
        d.x = s.nx;
        return d;
    }

    // The scene written into an array in whatever order `layout` names, through
    // the layout's own frame arithmetic -- so one helper produces a
    // phase-fastest file, a phase-slowest one and an angle-major one, and the
    // test cannot accidentally write the order it is trying to detect.
    Array5 storeAs(const Scene& s, const SimLayout& layout) {
        const Dims5 dims = dimsOf(s);
        const SimFrames frames = bindSimLayout(layout, dims);
        Array5 a = Array5::zeros(dims);
        for (Index angle = 0; angle < static_cast<Index>(s.nangles); ++angle)
            for (Index phase = 0; phase < static_cast<Index>(s.nphases); ++phase)
                for (Index z = 0; z < s.nz; ++z) {
                    const SimFrames::Frame f = frames.frameOf({angle, phase, z, 0, 0});
                    float* plane = a.plane(f.c, f.t, f.z);
                    for (Index y = 0; y < s.ny; ++y)
                        for (Index x = 0; x < s.nx; ++x)
                            plane[y * s.nx + x] = static_cast<float>(s.value(angle, phase, z, y, x));
                }
        return a;
    }

    DatasetMeta metaOf(const Scene& s, const std::string& layoutText = {}) {
        DatasetMeta m;
        m.name = "beads";
        m.sourcePath = "/not/read/beads.tif";
        m.format = "tiff";
        m.dims = dimsOf(s);
        m.sourceType = PixelType::UInt16;
        m.voxelUm = {kDxy, kDxy, kDz};
        ChannelInfo ch;
        ch.label = "StayGold";
        ch.wavelengthNm = 515.0;   // EMISSION: the file would be named 488 for the excitation line
        m.channels = {ch};
        if (!layoutText.empty()) m.sim = SimLayout::fromText(layoutText);
        return m;
    }

    // max |a - b| over every sample, relative to max |a|.
    double tableDiff(const OTFRadiallyAveraged& a, const OTFRadiallyAveraged& b) {
        const auto& da = a.data();
        const auto& db = b.data();
        for (int i = 0; i < 3; ++i)
            if (da.dimension(i) != db.dimension(i)) return 1.0;
        double worst = 0.0, peak = 0.0;
        for (Eigen::Index o = 0; o < da.dimension(0); ++o)
            for (Eigen::Index r = 0; r < da.dimension(1); ++r)
                for (Eigen::Index k = 0; k < da.dimension(2); ++k) {
                    peak = std::max(peak, std::abs(da(o, r, k)));
                    worst = std::max(worst, std::abs(da(o, r, k) - db(o, r, k)));
                }
        return peak > 0.0 ? worst / peak : worst;
    }

    std::string fieldsOf(const std::vector<OtfMeasureProblem>& problems) {
        std::string out;
        for (const OtfMeasureProblem& p : problems) out += p.field + " ";
        return out;
    }

    bool hasField(const std::vector<OtfMeasureProblem>& problems, const std::string& field) {
        return std::any_of(problems.begin(), problems.end(),
                           [&](const OtfMeasureProblem& p) { return p.field == field; });
    }

} // namespace

TEST_CASE("otfMeasureDefaults takes every number the acquisition states, and no more", "[otf_measure]") {
    Scene s;
    const DatasetMeta meta = metaOf(s, "z=[z, phase 3]");
    const OtfMeasureRequest r = otfMeasureDefaults(meta, dimsOf(s));

    CHECK(r.measure.dxy == kDxy);
    CHECK(r.measure.dz == kDz);
    CHECK(r.measure.nphases == 3);                                      // from the layout, not a guess
    CHECK(r.measure.packing == BeadPhasePacking::PhaseFastest);         // likewise
    CHECK(r.measure.detect.saturationLevel == 65535.0);                 // the pixel type's full scale

    // The three the acquisition genuinely does not state are left at the
    // library's defaults rather than invented from the data: a wrong bead size
    // or pattern period is a wrong table with nothing saying so.
    const OtfMeasureOptions lib;
    CHECK(r.measure.beadDiameterUm == lib.beadDiameterUm);
    CHECK(r.measure.patternPeriodUm == lib.patternPeriodUm);
    CHECK(r.measure.patternAngleRad == lib.patternAngleRad);

    // And no folder is proposed, so no caller can write a calibration artefact
    // into the acquisition tree by accepting a default.
    CHECK(r.path.empty());

    SECTION("a float stack has no full scale to clip at") {
        DatasetMeta f = meta;
        f.sourceType = PixelType::Float32;
        CHECK(otfMeasureDefaults(f, dimsOf(s)).measure.detect.saturationLevel == 0.0);
    }

    SECTION("without a layout the phase count stays the library's and the stack has to divide by it") {
        const DatasetMeta plain = metaOf(s);
        CHECK(otfMeasureDefaults(plain, dimsOf(s)).measure.nphases == lib.nphases);
    }
}

TEST_CASE("one validate answers for every front, and says which field each problem belongs to", "[otf_measure]") {
    Scene s;
    const DatasetMeta meta = metaOf(s);   // no layout: 36 sections on z
    const Dims5 dims = dimsOf(s);

    OtfMeasureRequest r = otfMeasureDefaults(meta, dims);
    r.channel = 3;                 // the dataset has one
    r.time = 2;                    // and one time point
    r.measure.nphases = 5;         // 36 sections do not divide by 5

    const std::vector<OtfMeasureProblem> problems = validateOtfMeasureRequest(r, meta, dims);
    INFO("fields: " << fieldsOf(problems));
    CHECK(problems.size() >= 3);
    CHECK(hasField(problems, "channel"));
    CHECK(hasField(problems, "time"));
    CHECK(hasField(problems, "measure"));
    for (const OtfMeasureProblem& p : problems) {
        CHECK_FALSE(p.field.empty());
        CHECK_FALSE(p.message.empty());
    }

    // The point of the whole struct: the library's condition arrives with the
    // LIBRARY's wording, not a second one invented here. 9k.50 item 5 was one
    // condition carrying three different strings across the three fronts.
    const std::string fromLibrary = validateOtfMeasure(r.measure, static_cast<int>(dims.z), static_cast<int>(dims.y),
                                                       static_cast<int>(dims.x));
    REQUIRE_FALSE(fromLibrary.empty());
    bool quoted = false;
    for (const OtfMeasureProblem& p : problems)
        if (p.field == "measure" && p.message == fromLibrary) quoted = true;
    CHECK(quoted);

    // One line for a caller with room for one.
    const std::string joined = otfMeasureProblems(problems);
    CHECK(joined.find("channel: ") != std::string::npos);
    CHECK(joined.find("; ") != std::string::npos);
    CHECK(otfMeasureProblems({}).empty());

    SECTION("a request the dataset fits has no problems at all") {
        CHECK(validateOtfMeasureRequest(otfMeasureDefaults(meta, dims), meta, dims).empty());
    }

    SECTION("and the run refuses it too, with the same words") {
        const Array5 a = storeAs(s, SimLayout::fromText("z=[z, phase 3]"));
        bool threw = false;
        try {
            measureOtfFromDataset(a, meta, r);
        } catch (const std::invalid_argument& e) {
            threw = true;
            CHECK(std::string(e.what()) == joined);
        }
        CHECK(threw);
    }
}

TEST_CASE("pixels that are not square are refused, with both numbers", "[otf_measure]") {
    Scene s;
    DatasetMeta meta = metaOf(s, "z=[z, phase 3]");
    meta.voxelUm = {0.0855, 0.1100, kDz};   // dx != dy
    const Dims5 dims = dimsOf(s);

    const std::vector<OtfMeasureProblem> problems = validateOtfMeasureRequest(otfMeasureDefaults(meta, dims), meta, dims);
    REQUIRE(hasField(problems, "dxy"));
    std::string message;
    for (const OtfMeasureProblem& p : problems)
        if (p.field == "dxy") message = p.message;
    // A radially averaged table has ONE radial step for both lateral axes, so
    // there is no dxy to pick; the message has to name what it saw.
    CHECK(message.find("0.0855") != std::string::npos);
    CHECK(message.find("0.11") != std::string::npos);

    SECTION("a one per cent difference is not refused") {
        meta.voxelUm = {0.0855, 0.0859, kDz};
        CHECK_FALSE(hasField(validateOtfMeasureRequest(otfMeasureDefaults(meta, dims), meta, dims), "dxy"));
    }
}

TEST_CASE("a raw-SIM layout that does not fit the array is named, not worked around", "[otf_measure]") {
    Scene s;
    DatasetMeta meta = metaOf(s);
    meta.sim = SimLayout::shorthand(3, 5);   // 15 per plane; 36 sections is not a multiple
    const Dims5 dims = dimsOf(s);

    const std::vector<OtfMeasureProblem> problems = validateOtfMeasureRequest(otfMeasureDefaults(meta, dims), meta, dims);
    REQUIRE(hasField(problems, "sim_layout"));
    for (const OtfMeasureProblem& p : problems)
        if (p.field == "sim_layout") {
            CHECK(p.message.find("36") != std::string::npos);      // what the file holds
            CHECK(p.message.find(dims.toString()) != std::string::npos);
        }
}

TEST_CASE("the phases are read through the layout, so a phase-slowest file measures the same table", "[otf_measure]") {
    Scene s;
    const Dims5 dims = dimsOf(s);

    const Array5 fastest = storeAs(s, SimLayout::fromText("z=[z, phase 3]"));
    const Array5 slowest = storeAs(s, SimLayout::fromText("z=[phase 3, z]"));
    // The two files really are different bytes; only the layout says so.
    bool differ = false;
    for (Index i = 0; i < dims.numel() && !differ; ++i) differ = fastest.data()[i] != slowest.data()[i];
    REQUIRE(differ);

    const DatasetMeta metaFast = metaOf(s, "z=[z, phase 3]");
    const DatasetMeta metaSlow = metaOf(s, "z=[phase 3, z]");
    const OtfMeasureReport a = measureOtfFromDataset(fastest, metaFast, otfMeasureDefaults(metaFast, dims));
    const OtfMeasureReport b = measureOtfFromDataset(slowest, metaSlow, otfMeasureDefaults(metaSlow, dims));

    CHECK(a.sectionOrder == OtfSectionOrder::Layout);
    CHECK(b.sectionOrder == OtfSectionOrder::Layout);
    CHECK(a.nphases == 3);
    CHECK(a.nz == 12);
    CHECK(a.sections == 36);
    // Same acquisition, two storage orders: one table, sample for sample.
    CHECK(tableDiff(a.measurement.otf, b.measurement.otf) == 0.0);
    CHECK(a.measurement.bandRatio == b.measurement.bandRatio);

    SECTION("and reading the phase-slowest file as the library's default order gives a different table") {
        // What this layer exists to prevent: with no layout stated, the z axis
        // is taken as it is stored, and the request's packing is believed.
        const DatasetMeta plain = metaOf(s);
        OtfMeasureRequest r = otfMeasureDefaults(plain, dims);
        REQUIRE(r.measure.packing == BeadPhasePacking::PhaseFastest);
        const OtfMeasureReport wrong = measureOtfFromDataset(slowest, plain, r);
        CHECK(wrong.sectionOrder == OtfSectionOrder::Packing);
        CHECK(tableDiff(b.measurement.otf, wrong.measurement.otf) > 1e-3);

        // Stating the right packing on the plain dataset recovers it exactly.
        r.measure.packing = BeadPhasePacking::PhaseSlowest;
        CHECK(tableDiff(b.measurement.otf, measureOtfFromDataset(slowest, plain, r).measurement.otf) == 0.0);
    }
}

TEST_CASE("a request that contradicts the layout is refused, not silently overridden", "[otf_measure]") {
    Scene s;
    const DatasetMeta meta = metaOf(s, "z=[z, phase 3]");
    const Dims5 dims = dimsOf(s);

    SECTION("the packing") {
        OtfMeasureRequest r = otfMeasureDefaults(meta, dims);
        r.measure.packing = BeadPhasePacking::PhaseSlowest;
        const std::vector<OtfMeasureProblem> problems = validateOtfMeasureRequest(r, meta, dims);
        REQUIRE(hasField(problems, "packing"));
    }
    SECTION("the phase count") {
        OtfMeasureRequest r = otfMeasureDefaults(meta, dims);
        r.measure.nphases = 1;
        const std::vector<OtfMeasureProblem> problems = validateOtfMeasureRequest(r, meta, dims);
        REQUIRE(hasField(problems, "nphases"));
        for (const OtfMeasureProblem& p : problems)
            if (p.field == "nphases") CHECK(p.message.find("3 phases") != std::string::npos);
    }
}

TEST_CASE("one direction is measured, and it is the one that was asked for", "[otf_measure]") {
    Scene s;
    s.nangles = 2;
    s.angleGain = 2.0;   // direction 1's bead is twice as bright
    const Dims5 dims = dimsOf(s);
    const DatasetMeta meta = metaOf(s, "z=[angle 2, z, phase 3]");
    const Array5 a = storeAs(s, SimLayout::fromText("z=[angle 2, z, phase 3]"));

    OtfMeasureRequest r0 = otfMeasureDefaults(meta, dims);
    OtfMeasureRequest r1 = r0;
    r1.angle = 1;
    const OtfMeasureReport m0 = measureOtfFromDataset(a, meta, r0);
    const OtfMeasureReport m1 = measureOtfFromDataset(a, meta, r1);

    CHECK(m0.angles == 2);
    CHECK(m0.sections == 36);      // one direction's frames, not all 72
    CHECK(m0.nz == 12);
    // The brighter direction has the larger zero-frequency sample, which is
    // what says the gather took the frames of the direction it was asked for.
    CHECK(m1.measurement.order0Dc > 1.5 * m0.measurement.order0Dc);
    CHECK(m1.provenance["stack"]["angle"].get<int>() == 1);
    CHECK(m0.provenance["stack"]["angles"].get<int>() == 2);

    SECTION("and a direction the acquisition does not hold is refused") {
        OtfMeasureRequest r2 = r0;
        r2.angle = 2;
        CHECK(hasField(validateOtfMeasureRequest(r2, meta, dims), "angle"));
    }
}

TEST_CASE("the run writes makeotf's layout, its sidecar and its provenance, and nothing without a path", "[otf_measure]") {
    Scene s;
    const Dims5 dims = dimsOf(s);
    const DatasetMeta meta = metaOf(s, "z=[z, phase 3]");
    const Array5 a = storeAs(s, SimLayout::fromText("z=[z, phase 3]"));

    SECTION("no path: measured, nothing written") {
        const OtfMeasureReport r = measureOtfFromDataset(a, meta, otfMeasureDefaults(meta, dims));
        CHECK(r.files.empty());
        CHECK(r.bytes == 0);
        CHECK(r.tablePath.empty());
        CHECK(r.measurement.nkr > 0);
        CHECK_FALSE(r.diagnostics.empty());
        CHECK(r.provenance["table"].is_null());
    }

    sirius::test::TempFile table("app_otf", "");   // no suffix: the service appends .tif
    OtfMeasureRequest r = otfMeasureDefaults(meta, dims);
    r.path = table.str;
    r.note = "a test";
    const OtfMeasureReport out = measureOtfFromDataset(a, meta, r);

    const std::filesystem::path tif(table.str + ".tif");
    REQUIRE(out.tablePath == tif);
    REQUIRE(out.files.size() == 3);
    CHECK(out.files[0] == tif.string());
    CHECK(out.files[1] == tif.string() + ".toml");
    CHECK(out.files[2] == tif.string() + ".json");
    CHECK(out.bytes > 0);
    for (const std::string& f : out.files) CHECK(std::filesystem::exists(f));

    // makeotf's own layout: norders pages of nkr rows by 2 * nzotf columns,
    // kz fastest, real and imaginary interleaved.
    const ImageStack<float> pages = readTiffStack<float>(tif.string());
    CHECK(pages.dimension(0) == out.measurement.norders);
    CHECK(pages.dimension(1) == out.measurement.nkr);
    CHECK(pages.dimension(2) == 2 * out.measurement.nzotf);
    // Order 0's zero-frequency sample is 1 under the default scale, which is
    // what loadOTF's absolute cutoff and the Wiener constant are calibrated to.
    CHECK(std::abs(pages(0, 0, 0) - 1.0f) < 1e-5f);
    CHECK(std::abs(pages(0, 0, 1)) < 1e-5f);

    std::ostringstream side;
    {
        std::ifstream in(tif.string() + ".toml");
        REQUIRE(in);
        side << in.rdbuf();
    }
    const std::string sidecar = side.str();
    CHECK(sidecar.find("[sampling]") != std::string::npos);
    CHECK(sidecar.find("dkr = ") != std::string::npos);
    CHECK(sidecar.find("kz_origin") != std::string::npos);
    CHECK(sidecar.find("a test") != std::string::npos);

    for (const std::string& f : out.files) std::filesystem::remove(f);

    SECTION("an existing table is refused unless overwriting was asked for") {
        sirius::test::TempFile taken("app_otf_taken", ".tif");
        { std::ofstream(taken.str) << "not an OTF"; }
        OtfMeasureRequest again = otfMeasureDefaults(meta, dims);
        again.path = taken.str;
        CHECK(hasField(validateOtfMeasureRequest(again, meta, dims), "path"));
        again.overwrite = true;
        CHECK_FALSE(hasField(validateOtfMeasureRequest(again, meta, dims), "path"));
        const OtfMeasureReport ow = measureOtfFromDataset(a, meta, again);
        for (const std::string& f : ow.files) std::filesystem::remove(f);
    }

    SECTION("a folder that does not exist is refused before anything is measured") {
        OtfMeasureRequest bad = otfMeasureDefaults(meta, dims);
        bad.path = (std::filesystem::temp_directory_path() / "sirius-no-such-folder-9k56" / "otf.tif").string();
        CHECK(hasField(validateOtfMeasureRequest(bad, meta, dims), "path"));
    }
}

TEST_CASE("the provenance ties the table to the acquisition it came from", "[otf_measure]") {
    Scene s;
    const Dims5 dims = dimsOf(s);
    const DatasetMeta meta = metaOf(s, "z=[z, phase 3]");
    const Array5 a = storeAs(s, SimLayout::fromText("z=[z, phase 3]"));
    const OtfMeasureReport out = measureOtfFromDataset(a, meta, otfMeasureDefaults(meta, dims));
    const nlohmann::json& p = out.provenance;

    // which stack
    CHECK(p["stack"]["path"].get<std::string>() == meta.sourcePath);
    CHECK(p["stack"]["name"].get<std::string>() == "beads");
    CHECK(p["stack"]["sim_layout"].get<std::string>() == meta.sim.text());
    CHECK(p["stack"]["section_order"].get<std::string>() == "layout");
    CHECK(p["stack"]["sections"].get<int>() == 36);
    CHECK(p["stack"]["nphases"].get<int>() == 3);
    CHECK(p["stack"]["channel_emission_nm"].get<double>() == 515.0);
    CHECK(p["stack"]["source_type"].get<std::string>() == "uint16");

    // which parameters
    CHECK(p["options"]["dxy_um"].get<double>() == kDxy);
    CHECK(p["options"]["scale"].get<std::string>() == "order0_dc");
    CHECK(p["options"]["packing"].get<std::string>() == "phase_fastest");
    CHECK(p["options"]["bead_diameter_um"].get<double>() == OtfMeasureOptions{}.beadDiameterUm);
    // the line spacing the side-band division used, and whether anyone stated
    // it: a table measured on makeotf's 0.2 um default cannot be told from one
    // measured on the instrument's own by looking at the samples, so the
    // record has to say. This dataset states none.
    CHECK(p["options"]["pattern_period_um"].get<double>() == 0.0);
    CHECK_FALSE(p["options"]["pattern_period_stated"].get<bool>());
    CHECK(p["options"]["pattern_period_um_used"].get<double>() == sirius::kMakeotfLineSpacingUm);
    bool warned = false;
    for (const std::string& w : out.diagnostics.warnings)
        warned = warned || w.find("line spacing was not stated") != std::string::npos;
    CHECK(warned);

    // which beads, and what was rejected
    REQUIRE(p["beads"].size() == out.measurement.beads.size());
    REQUIRE(out.measurement.kept >= 1);
    std::size_t kept = 0, listed = 0;
    for (const nlohmann::json& b : p["beads"]) {
        if (b["kept"].get<bool>()) ++kept;
        ++listed;
    }
    CHECK(static_cast<int>(kept) == out.measurement.kept);
    CHECK(listed == out.measurement.beads.size());
    for (std::size_t i = 0; i < out.measurement.beads.size(); ++i)
        CHECK(p["beads"][i]["status"].get<std::string>() ==
              std::string(beadRejectionName(out.measurement.beads[i].rejection)));
    CHECK(p.contains("rejected"));

    // the sampling, so the table is never read at whatever pixel size a later
    // run happens to use (findings 9k.48)
    CHECK(p["sampling"]["dkr_per_um"].get<double>() == out.measurement.dkr);
    CHECK(p["sampling"]["dkz_per_um"].get<double>() == out.measurement.dkz);
    CHECK(p["sampling"]["kz_origin"].get<std::string>() == "dc_first");
    CHECK(p["sampling"]["nkr"].get<int>() == out.measurement.nkr);

    // and how soft the scale is, which is the number to read before the depth
    CHECK(p["scale"]["divisor"].get<double>() == out.measurement.scaleDivisor);
    CHECK(p["result"]["band_ratio"].get<double>() == out.measurement.bandRatio);
    CHECK_FALSE(p["sirius"]["commit"].get<std::string>().empty());

    // the service's own decisions are recorded, not only taken
    REQUIRE_FALSE(out.notes.empty());
    bool saysLayout = false;
    for (const std::string& n : out.notes) saysLayout = saysLayout || n.find("gathered through") != std::string::npos;
    CHECK(saysLayout);
}

TEST_CASE("the diagnostics are the dock's own types, with the marks in the preview's pixels", "[otf_measure]") {
    Scene s;
    const Dims5 dims = dimsOf(s);
    const DatasetMeta meta = metaOf(s, "z=[z, phase 3]");
    const Array5 a = storeAs(s, SimLayout::fromText("z=[z, phase 3]"));
    OtfMeasureRequest r = otfMeasureDefaults(meta, dims);
    r.previewMaxSide = 32;   // smaller than the 64 px section, so the scaling is exercised
    const OtfMeasureReport out = measureOtfFromDataset(a, meta, r);
    const Diagnostics& d = out.diagnostics;

    // Generic, because the Sim body expects three spectra per tab and four
    // named tabs; the dialog draws the table itself with cells::table.
    CHECK(d.kind == DiagnosticsKind::Generic);
    CHECK_FALSE(d.empty());
    REQUIRE(d.images.size() == static_cast<std::size_t>(1 + out.measurement.norders));
    REQUIRE_FALSE(d.tabs.empty());
    CHECK(d.tabs.front().images.size() == 2);   // the field, then order 0

    const DiagnosticImage& preview = d.images[0];
    CHECK(preview.rows > 0);
    CHECK(preview.cols > 0);
    CHECK(preview.cols <= 32);
    CHECK(preview.values.size() == static_cast<std::size_t>(preview.rows * preview.cols));
    REQUIRE_FALSE(preview.marks.empty());
    for (const DiagnosticMark& m : preview.marks) {
        // diagnostic_cells.cpp scales marks by the texture's size, so they are
        // in the thumbnail's pixels and not the stack's.
        CHECK(m.x >= 0.0);
        CHECK(m.x <= static_cast<double>(preview.cols));
        CHECK(m.y >= 0.0);
        CHECK(m.y <= static_cast<double>(preview.rows));
    }

    const DiagnosticImage& order0 = d.images[1];
    CHECK(order0.logScale);
    CHECK(order0.rows == out.measurement.nkr);
    CHECK(order0.cols == out.measurement.nzotf);

    REQUIRE(d.curves.size() == static_cast<std::size_t>(out.measurement.norders + 1));
    CHECK(d.curves[0].x.size() == static_cast<std::size_t>(out.measurement.nkr));
    CHECK(d.curves[0].y.size() == d.curves[0].x.size());
    CHECK(d.curves.back().x.size() == d.curves.back().y.size());

    REQUIRE(d.table.has_value());
    CHECK(d.table->header.size() == 9);
    CHECK(d.table->rows.size() == out.measurement.beads.size());
    for (const std::vector<std::string>& row : d.table->rows) CHECK(row.size() == d.table->header.size());
    CHECK(d.table->accentCells.size() == static_cast<std::size_t>(out.measurement.kept));

    CHECK(d.facts.size() >= 10);
    CHECK_FALSE(d.summary.empty());
    CHECK(d.footer == out.measurement.summary());
    CHECK(out.summary == d.summary);

    SECTION("and a report can be redrawn from the result alone") {
        const Diagnostics again = otfMeasureDiagnostics(out.measurement, r);
        CHECK(again.images.size() == static_cast<std::size_t>(out.measurement.norders));   // no preview given
        CHECK(again.summary == d.summary);
        CHECK(again.table->rows.size() == d.table->rows.size());
    }
}

TEST_CASE("a field of beads reports every candidate, kept or not, with its reason", "[otf_measure]") {
    Scene s;
    // Two beads the detector must see as two. Its non-maximum suppression is a
    // BOX of half-width round(minSeparationLateralUm / dxy) = 12 px
    // (src/otf_measure.cpp), so beads 12 px apart in EVERY axis leave one
    // candidate however far apart they are in Euclidean distance -- which is
    // what my first placement, (26, 26) and (38, 38), measured: found = 1. And
    // the field has to be large enough for the boundary filter, which needs
    // roiLateral/2 + boundaryMargin = 21 px of clearance on every side.
    s.ny = s.nx = 96;
    s.beads = {{32.0, 48.0, 6.0, 3000.0}, {64.0, 48.0, 6.0, 2400.0}};
    const Dims5 dims = dimsOf(s);
    const DatasetMeta meta = metaOf(s, "z=[z, phase 3]");
    const Array5 a = storeAs(s, SimLayout::fromText("z=[z, phase 3]"));

    OtfMeasureRequest r = otfMeasureDefaults(meta, dims);
    r.measure.field = true;
    const OtfMeasureReport out = measureOtfFromDataset(a, meta, r);

    CHECK(out.measurement.found >= 2);
    CHECK(out.measurement.kept >= 1);
    REQUIRE_FALSE(out.measurement.beads.empty());
    for (const BeadFit& b : out.measurement.beads) CHECK(b.kept == (b.rejection == BeadRejection::None));

    // every candidate in the inventory is in the table and in the provenance,
    // with one status wording -- the library's beadRejectionName
    REQUIRE(out.diagnostics.table.has_value());
    REQUIRE(out.diagnostics.table->rows.size() == out.measurement.beads.size());
    for (std::size_t i = 0; i < out.measurement.beads.size(); ++i)
        CHECK(out.diagnostics.table->rows[i].back() ==
              std::string(beadRejectionName(out.measurement.beads[i].rejection)));

    // The counts have to add up to what the detector did: every candidate it
    // found is in exactly one bucket (the ones the amplitude floor and the
    // examination cap cut short included), and the bucket whose name is
    // "kept" holds exactly the kept ones. Before BeadRejection gained
    // OverMaxBeads and NotExamined, neither was true.
    int counted = 0;
    for (const nlohmann::json& n : out.provenance["rejected"]) counted += n.get<int>();
    CHECK(counted == out.measurement.found);
    CHECK(counted >= static_cast<int>(out.measurement.beads.size()));
    CHECK(out.provenance["rejected"].value("kept", 0) == out.measurement.kept);

    SECTION("a single-bead run reports the one centre makeotf would have taken") {
        OtfMeasureRequest one = otfMeasureDefaults(meta, dims);
        one.measure.field = false;
        const OtfMeasureReport single = measureOtfFromDataset(a, meta, one);
        CHECK(single.measurement.beads.size() == 1);
        CHECK(single.measurement.kept == 1);
        CHECK(single.diagnostics.table->rows.size() == 1);
        CHECK(single.diagnostics.table->caption == "Bead centre");
    }
}

TEST_CASE("the JSON face parses once for every front, and an unknown enum says what it accepts", "[otf_measure]") {
    Scene s;
    const Dims5 dims = dimsOf(s);
    const DatasetMeta meta = metaOf(s, "z=[z, phase 3]");

    SECTION("what is absent keeps the dataset's answer") {
        const OtfMeasureRequest r = otfMeasureRequestFromJson(nlohmann::json::object(), meta, dims);
        const OtfMeasureRequest d = otfMeasureDefaults(meta, dims);
        CHECK(r.measure.dxy == d.measure.dxy);
        CHECK(r.measure.nphases == d.measure.nphases);
        CHECK(r.measure.packing == d.measure.packing);
        CHECK(r.measure.detect.saturationLevel == d.measure.detect.saturationLevel);
        CHECK(r.path.empty());
        CHECK(otfMeasureRequestFromJson(nlohmann::json(), meta, dims).measure.dxy == d.measure.dxy);
    }

    SECTION("and what is present is applied, nested options included") {
        const nlohmann::json args = {{"path", "/tmp/otf.tif"},
                                     {"angle", 0},
                                     {"field", true},
                                     {"scale", "as_measured"},
                                     {"background_estimate", "darkest_fraction"},
                                     {"bead_diameter_um", 0.1},
                                     {"bead_compensation_pixel_um", 0.106},
                                     {"detect", {{"max_beads", 9}, {"min_amplitude", 510.8}}}};
        const OtfMeasureRequest r = otfMeasureRequestFromJson(args, meta, dims);
        CHECK(r.path == "/tmp/otf.tif");
        CHECK(r.measure.field);
        CHECK(r.measure.scale == OtfMeasureScale::AsMeasured);
        CHECK(r.measure.backgroundEstimate == BackgroundEstimate::DarkestFraction);
        CHECK(r.measure.beadDiameterUm == 0.1);
        CHECK(r.measure.beadCompensationPixelUm == 0.106);
        CHECK(r.measure.detect.maxBeads == 9);
        CHECK(r.measure.detect.minAmplitude == 510.8);
        // untouched keys still come from the dataset
        CHECK(r.measure.dxy == kDxy);
        CHECK(r.measure.detect.roiLateralUm == BeadDetectionOptions{}.roiLateralUm);
    }

    SECTION("a key this face does not read is refused, not dropped") {
        // Dropping is the worse of the two failures: "patternPeriodUm" or
        // "pattern_period" in a tool call used to measure on the default line
        // spacing and report success, which is a wrong constant with nothing
        // saying so.
        for (const char* key : {"patternPeriodUm", "pattern_period", "bead_diameter", "dataset"}) {
            nlohmann::json args = nlohmann::json::object();
            args[key] = 0.5;
            bool threw = false;
            try {
                otfMeasureRequestFromJson(args, meta, dims);
            } catch (const std::invalid_argument& e) {
                threw = true;
                CHECK(std::string(e.what()).find(key) != std::string::npos);
            }
            CHECK(threw);
        }
        // the same inside the nested object, and a detect that is not one
        nlohmann::json nested = {{"detect", {{"maxBeads", 3}}}};
        CHECK_THROWS_AS(otfMeasureRequestFromJson(nested, meta, dims), std::invalid_argument);
        nlohmann::json notAnObject = {{"detect", 3}};
        CHECK_THROWS_AS(otfMeasureRequestFromJson(notAnObject, meta, dims), std::invalid_argument);
        // and every key the header documents is still accepted
        const nlohmann::json every = {
            {"channel", 0}, {"time", 0}, {"angle", 0}, {"path", ""}, {"note", ""}, {"overwrite", false},
            {"sidecar", true}, {"provenance", true}, {"preview_max_side", 64}, {"nphases", 3},
            {"norders", 2}, {"packing", "phase_fastest"}, {"phases", nlohmann::json::array({0.0, 1.0, 2.0})},
            {"dxy", 0.1}, {"dz", 0.2}, {"background", -1.0}, {"background_estimate", "border_mean"},
            {"background_border", 6}, {"darkest_fraction", 0.1}, {"apodize", 4}, {"bead_diameter_um", 0.12},
            {"pattern_period_um", 0.504}, {"pattern_angle_rad", 1.57}, {"bead_compensation_pixel_um", 0.0},
            {"bead_compensation_axial_um", 0.0}, {"scale", "order0_dc"}, {"line_fit_first", 2},
            {"line_fit_last", 9}, {"band_ratio_min_order0", 0.02}, {"repair_kr0_column", true},
            {"combine_reim", true}, {"field", false}, {"per_bead_normalise", true},
            {"detect", {{"dog_small_lateral_um", 0.1}, {"dog_small_axial_um", 0.1},
                        {"dog_large_lateral_um", 5.0}, {"dog_large_axial_um", 5.0},
                        {"min_separation_lateral_um", 1.0}, {"min_separation_axial_um", 0.0},
                        {"min_amplitude", 0.0}, {"min_amplitude_fraction", 0.05}, {"saturation_level", 0.0},
                        {"roi_lateral_um", 1.5}, {"roi_axial_um", 0.0}, {"boundary_margin_lateral_um", 1.0},
                        {"boundary_margin_axial_um", 0.0}, {"sigma_min_lateral_um", 0.05},
                        {"sigma_max_lateral_um", 0.2}, {"sigma_min_axial_um", 0.0},
                        {"sigma_max_axial_um", 0.0}, {"max_beads", 64}, {"max_residual", 0.0}}}};
        const OtfMeasureRequest r = otfMeasureRequestFromJson(every, meta, dims);
        CHECK(r.measure.patternPeriodUm == 0.504);
        CHECK(r.measure.detect.maxBeads == 64);
    }

    SECTION("an enum value no front end should have sent names the key and its values, once") {
        for (const char* key : {"packing", "scale", "background_estimate"}) {
            nlohmann::json args = nlohmann::json::object();
            args[key] = "sideways";
            bool threw = false;
            try {
                otfMeasureRequestFromJson(args, meta, dims);
            } catch (const std::invalid_argument& e) {
                threw = true;
                const std::string what = e.what();
                CHECK(what.find(key) != std::string::npos);
                CHECK(what.find("sideways") != std::string::npos);
            }
            CHECK(threw);
        }
    }

    SECTION("the reply carries what was written, the sampling, and both depth numbers") {
        const Array5 a = storeAs(s, SimLayout::fromText("z=[z, phase 3]"));
        const OtfMeasureReport out = measureOtfFromDataset(a, meta, otfMeasureDefaults(meta, dims));
        const nlohmann::json reply = otfMeasureReportJson(out);
        CHECK(reply["table"].is_null());               // nothing written without a path
        CHECK(reply["files"].empty());
        CHECK(reply["section_order"].get<std::string>() == "layout");
        CHECK(reply["sections"].get<int>() == 36);
        CHECK(reply["otf"]["nkr"].get<int>() == out.measurement.nkr);
        CHECK(reply["otf"]["dkr_per_um"].get<double>() == out.measurement.dkr);
        CHECK(reply["otf"]["kz_origin"].get<std::string>() == "dc_first");
        CHECK(reply["beads"]["kept"].get<int>() == out.measurement.kept);
        CHECK(reply["band_ratio"].get<double>() == out.measurement.bandRatio);
        CHECK(reply["modulation_depth"].get<double>() == out.measurement.modulationDepth);
        CHECK(reply["scale"]["used"].get<std::string>() == "order0_dc");
        CHECK(reply["provenance"]["stack"]["sections"].get<int>() == 36);
        CHECK_FALSE(reply["warnings"].empty());
        CHECK(reply["summary"].get<std::string>() == out.summary);
    }
}

TEST_CASE("otfMeasureFileName names the channel and proposes no folder", "[otf_measure]") {
    Scene s;
    DatasetMeta meta = metaOf(s);
    CHECK(otfMeasureFileName(meta, 0) == "OTF_515_sirius.tif");   // the EMISSION the channel carries

    meta.channels[0].wavelengthNm = 0.0;
    meta.channels[0].label = "mito/gfp 2";
    const std::string named = otfMeasureFileName(meta, 0);
    CHECK(named.find('/') == std::string::npos);
    CHECK(named.find('\\') == std::string::npos);
    CHECK(named.find(' ') == std::string::npos);

    CHECK(otfMeasureFileName(DatasetMeta{}, 0) == "OTF_sirius.tif");
    CHECK(otfMeasureFileName(meta, 7) == "OTF_sirius.tif");       // a channel the dataset does not have
}

// --- the user's own sparse bead field, when it is at hand --------------------
// Everything above is synthetic, because the service's job is the layer
// between a dataset and the measurement and a synthetic scene is the only one
// whose answer is known. This case is the other half: the SAME acquisition the
// library's own real-data case uses (the sparse bead field, never the dense
// one -- the user ruled the dense field out), driven through the app service
// from an Array5 and a DatasetMeta rather than from a bare tensor, and
// compared with the colleague's makeotf table, which IS committed
// (tests/data/isoar2_sparse_OTF_488.tif). Only the 4.6 MB raw stack is not, so
// the environment names it -- SIRIUS_OTF_BEAD_STACK, the same variable
// tests/test_otf_measure.cpp uses, so one job drives both.
//
// The claim is exact rather than approximate: the file is uint16, which float
// holds without loss, so going through Array5's float32 cannot move a sample
// and this table has to be the one the library measured.
TEST_CASE("the app service measures the user's own sparse bead field", "[otf_measure][real]") {
    const char* stackPath = std::getenv("SIRIUS_OTF_BEAD_STACK");
    if (stackPath == nullptr || *stackPath == '\0')
        SKIP("set SIRIUS_OTF_BEAD_STACK to the sparse bead field's raw stack");
    const char* refEnv = std::getenv("SIRIUS_OTF_BEAD_REFERENCE");
    const std::string refPath = refEnv && *refEnv ? std::string(refEnv)
                                                  : std::string(SIRIUS_TEST_DATA_DIR) + "/isoar2_sparse_OTF_488.tif";

    const ImageStack<float> raw = readTiffStack<float>(stackPath);
    REQUIRE(raw.dimension(0) > 0);
    Dims5 dims;
    dims.c = dims.t = 1;
    dims.z = static_cast<Index>(raw.dimension(0));
    dims.y = static_cast<Index>(raw.dimension(1));
    dims.x = static_cast<Index>(raw.dimension(2));
    Array5 a = Array5::zeros(dims);
    std::copy(raw.data(), raw.data() + dims.numel(), a.data());

    DatasetMeta meta;
    meta.name = "RAW_488_3phase_ols20px_3G";
    meta.sourcePath = stackPath;
    meta.format = "tiff";
    meta.dims = dims;
    meta.sourceType = PixelType::UInt16;
    // the acquisition's own numbers: the camera pixel and the SampleMotion
    // step out of 488_3phase_ols20px_3G_JSONsettings.json
    meta.voxelUm = {0.085526315789473686, 0.085526315789473686, 0.1};
    ChannelInfo ch;
    ch.label = "StayGold";
    ch.wavelengthNm = 515.0;
    meta.channels = {ch};
    // phase fastest, which the acquisition's slicelist.sqlite3 states
    // independently: three rows share each Slice_Index
    meta.sim = SimLayout::fromText("z=[z, phase 3]");

    OtfMeasureRequest r = otfMeasureDefaults(meta, dims);
    REQUIRE(validateOtfMeasureRequest(r, meta, dims).empty());
    CHECK(r.measure.nphases == 3);
    CHECK(r.measure.dxy == 0.085526315789473686);
    CHECK(r.measure.dz == 0.1);
    // makeotf's own defaults for what the colleague did not pass, and its
    // default dr of 0.106 um in the finite-bead-size division -- the only step
    // that reads a pixel size -- which is what reproducing their table needs
    r.measure.beadDiameterUm = 0.12;
    r.measure.patternPeriodUm = 0.2;
    r.measure.patternAngleRad = 1.57;
    r.measure.beadCompensationPixelUm = 0.106;

    const OtfMeasureReport out = measureOtfFromDataset(a, meta, r);
    INFO("measured: " << out.measurement.summary());
    CHECK(out.sectionOrder == OtfSectionOrder::Layout);
    CHECK(out.nphases == 3);
    CHECK(out.nz == 101);
    CHECK(out.measurement.norders == 2);
    CHECK(out.measurement.nkr == 65);
    CHECK(out.measurement.nzotf == 101);
    CHECK(out.measurement.hermitianKzError == 0.0);
    CHECK(std::abs(out.measurement.dkr - 0.09134615384615385) < 1e-12);
    CHECK(std::abs(out.measurement.dkz - 0.09900990099009901) < 1e-12);
    CHECK(out.provenance["sampling"]["dkr_per_um"].get<double>() == out.measurement.dkr);
    CHECK(out.provenance["stack"]["section_order"].get<std::string>() == "layout");

    // the colleague's table, read in makeotf's own layout: norders pages of
    // nkr rows by 2 * nzotf columns, kz fastest, re/im interleaved
    const ImageStack<float> ref = readTiffStack<float>(refPath);
    REQUIRE(ref.dimension(0) == out.measurement.norders);
    REQUIRE(ref.dimension(1) == out.measurement.nkr);
    REQUIRE(ref.dimension(2) == 2 * out.measurement.nzotf);

    // ONE global real scale, which is the only quantity the two runs can
    // legitimately differ in: makeotf divides every order by order 0's
    // zero-frequency sample, and that sample is the integral of the
    // background-subtracted band, so it follows the background estimate.
    // Order 0's (0, 0) sample is 1 in both tables by construction and is left
    // out of the fit.
    const auto& mine = out.measurement.otf.data();
    double num = 0.0, den = 0.0, refPeak = 0.0;
    for (Eigen::Index o = 0; o < mine.dimension(0); ++o)
        for (Eigen::Index q = 0; q < mine.dimension(1); ++q)
            for (Eigen::Index k = 0; k < mine.dimension(2); ++k) {
                if (o == 0 && q == 0 && k == 0) continue;
                const double rr = ref(o, q, 2 * k), ri = ref(o, q, 2 * k + 1);
                num += rr * mine(o, q, k).real() + ri * mine(o, q, k).imag();
                den += mine(o, q, k).real() * mine(o, q, k).real() + mine(o, q, k).imag() * mine(o, q, k).imag();
                refPeak = std::max(refPeak, std::hypot(rr, ri));
            }
    REQUIRE(den > 0.0);
    REQUIRE(refPeak > 0.0);
    const double scale = num / den;
    // The DC is left out of the RESIDUAL as well as the fit, and it has to be:
    // both tables are divided by their own order-0 DC, so that sample is 1 in
    // both WHATEVER the scale is, and multiplying one table by s moves it off
    // 1 by construction. My first version of this loop included it and
    // reported a residual of 1.49 of peak, which was the DC alone
    // (|0.4734 - 1| / 0.3531) and not a disagreement about any measured
    // sample. That the rest then matches to float rounding under ONE scale is
    // the structural fact the library's header states: the background enters
    // only the kx = ky = 0 column, and modify() has replaced that column
    // everywhere but order 0's kz = 0 -- so the background estimate can move
    // exactly one sample of the raw table, the DC, and therefore nothing but
    // the scale every order is divided by.
    double worst = 0.0;
    for (Eigen::Index o = 0; o < mine.dimension(0); ++o)
        for (Eigen::Index q = 0; q < mine.dimension(1); ++q)
            for (Eigen::Index k = 0; k < mine.dimension(2); ++k) {
                if (o == 0 && q == 0 && k == 0) continue;
                worst = std::max(worst, std::hypot(scale * mine(o, q, k).real() - ref(o, q, 2 * k),
                                                   scale * mine(o, q, k).imag() - ref(o, q, 2 * k + 1)));
            }
    INFO("one global scale " << scale << ", worst residual " << worst / refPeak << " of peak");
    CHECK(worst / refPeak < 1e-5);
    // and the one sample the scale cannot carry is 1 in both
    CHECK(std::abs(mine(0, 0, 0).real() - 1.0) < 1e-6);
    CHECK(std::abs(ref(0, 0, 0) - 1.0f) < 1e-6f);

    // And the number that scale cannot move, which is what to quote off a
    // table: the reference's own band ratio, computed the same way.
    CHECK(out.measurement.bandRatio > 0.0);
    CHECK(out.measurement.bandRatioSamples > 100);
}
