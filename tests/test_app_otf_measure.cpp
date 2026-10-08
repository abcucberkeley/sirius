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

    int counted = 0;
    for (const nlohmann::json& n : out.provenance["rejected"]) counted += n.get<int>();
    CHECK(counted >= static_cast<int>(out.measurement.beads.size()));

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
