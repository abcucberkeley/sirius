// Training data export: the class table, the 3D and per plane bounding
// boxes, the semantic mask, the YOLO text, and the folder an export leaves
// behind -- which has to accumulate samples rather than replace them, since
// the point is to build a dataset out of many runs.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <sirius/tiff_io.hpp>

#include "core/training_export.hpp"

using namespace sirius;
using namespace sirius::app;

namespace {

    struct TempDir {
        std::filesystem::path dir;
        TempDir() {
            dir = std::filesystem::temp_directory_path() /
                  ("sirius-training-" + std::to_string(std::filesystem::hash_value(std::filesystem::temp_directory_path())) + "-" +
                   std::to_string(reinterpret_cast<std::uintptr_t>(this)));
            std::filesystem::remove_all(dir);
            std::filesystem::create_directories(dir);
        }
        ~TempDir() {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
    };

    // Two cubes at opposite corners plus one voxel of dust.
    LabelVolume twoCubes(Index t = 1) {
        LabelVolume labels(t, 6, 8, 10);
        for (Index k = 0; k < t; ++k) {
            std::uint32_t* v = labels.volume(k);
            auto put = [&](Index z, Index y, Index x, std::uint32_t id) { v[(z * 8 + y) * 10 + x] = id; };
            for (Index z = 1; z <= 2; ++z)
                for (Index y = 1; y <= 2; ++y)
                    for (Index x = 1; x <= 2; ++x) put(z, y, x, 1);
            for (Index z = 3; z <= 4; ++z)
                for (Index y = 5; y <= 6; ++y)
                    for (Index x = 6; x <= 8; ++x) put(z, y, x, 2);
            put(0, 0, 0, 3);   // dust, and it touches the border
        }
        labels.recomputeStats(0);
        return labels;
    }

    Array5 image(const LabelVolume& labels) {
        Array5 a = Array5::zeros(Dims5{1, labels.t(), labels.z(), labels.y(), labels.x()});
        for (Index t = 0; t < labels.t(); ++t)
            for (Index i = 0, n = labels.volumeSize(); i < n; ++i) a.volume(0, t).data()[i] = labels.volume(t)[i] > 0 ? 1000.0f : 10.0f;
        return a;
    }

    DatasetMeta meta() {
        DatasetMeta m;
        m.voxelUm = {0.1, 0.1, 0.2};
        m.sourcePath = "synthetic.tif";
        return m;
    }

} // namespace

TEST_CASE("Bounding boxes describe every object", "[app][training]") {
    LabelVolume labels = twoCubes();
    const ClassTable classes = classTable(labels);
    REQUIRE(classes.names == std::vector<std::string>{"object"});
    CHECK(classes.idOf("object") == 1);
    CHECK(classes.idOf("nothing") == 0);
    CHECK(classes.nameOf(1) == "object");
    CHECK(classes.nameOf(0) == "background");
    CHECK(classes.nameOf(7) == "background");

    const std::vector<LabelBox> boxes = boundingBoxes(labels, 0, classes, 1);
    REQUIRE(boxes.size() == 3);
    // sorted by label id, so the order does not depend on the scan
    CHECK(boxes[0].label == 1);
    CHECK(boxes[1].label == 2);
    CHECK(boxes[2].label == 3);

    // half open, like LabelStats::bbox
    CHECK(boxes[0].bbox == std::array<Index, 6>{1, 3, 1, 3, 1, 3});
    CHECK(boxes[0].voxels == 8);
    CHECK(boxes[0].dz() == 2);
    CHECK(boxes[0].dy() == 2);
    CHECK(boxes[0].dx() == 2);
    CHECK_FALSE(boxes[0].touchesBorder);
    CHECK_THAT(boxes[0].cz, Catch::Matchers::WithinAbs(1.5, 1e-9));
    CHECK_THAT(boxes[0].cx, Catch::Matchers::WithinAbs(1.5, 1e-9));

    CHECK(boxes[1].bbox == std::array<Index, 6>{3, 5, 5, 7, 6, 9});
    CHECK(boxes[1].voxels == 12);
    CHECK_FALSE(boxes[1].touchesBorder);   // z 3..4 of 6, y 5..6 of 8, x 6..8 of 10: clear on every side
    CHECK(boxes[2].label == 3);
    CHECK(boxes[2].touchesBorder);   // the dust sits in the corner

    CHECK(boxes[0].classId == 1);
    CHECK(boxes[0].className == "object");

    SECTION("small objects can be left out") {
        const std::vector<LabelBox> big = boundingBoxes(labels, 0, classes, 8);
        REQUIRE(big.size() == 2);
        CHECK(big[0].label == 1);
        CHECK(big[1].label == 2);
    }

    SECTION("a time point that does not exist gives nothing") {
        CHECK(boundingBoxes(labels, 5, classes, 1).empty());
        CHECK(boundingBoxes(labels, -1, classes, 1).empty());
    }
}

TEST_CASE("Classes come from the label statistics", "[app][training]") {
    LabelVolume labels = twoCubes();
    labels.stats()[0].cls = "nucleus";
    labels.stats()[1].cls = "filament";
    const ClassTable classes = classTable(labels);
    // sorted, so ids do not depend on which label was reviewed first
    REQUIRE(classes.names == std::vector<std::string>{"filament", "nucleus", "object"});
    const std::vector<LabelBox> boxes = boundingBoxes(labels, 0, classes, 1);
    REQUIRE(boxes.size() == 3);
    CHECK(boxes[0].className == "nucleus");
    CHECK(boxes[0].classId == 2);
    CHECK(boxes[1].className == "filament");
    CHECK(boxes[1].classId == 1);

    const Buffer<std::uint8_t> semantic = semanticVolume(labels, 0, classes);
    CHECK(semantic.data()[(1 * 8 + 1) * 10 + 1] == 2);   // inside the nucleus
    CHECK(semantic.data()[(3 * 8 + 5) * 10 + 6] == 1);   // inside the filament
    CHECK(semantic.data()[(0 * 8 + 4) * 10 + 4] == 0);   // background stays 0
}

TEST_CASE("Per plane boxes and YOLO text", "[app][training]") {
    LabelVolume labels = twoCubes();
    const ClassTable classes = classTable(labels);
    const std::vector<SliceBox> slices = sliceBoxes(labels, 0, classes, 1);
    // cube 1 on z1 and z2, cube 2 on z3 and z4, dust on z0
    CHECK(slices.size() == 5);
    const auto onZ = [&](Index z) {
        std::vector<SliceBox> out;
        for (const SliceBox& s : slices)
            if (s.z == z) out.push_back(s);
        return out;
    };
    CHECK(onZ(0).size() == 1);
    CHECK(onZ(1).size() == 1);
    CHECK(onZ(5).empty());
    const std::vector<SliceBox> z1 = onZ(1);
    CHECK(z1[0].label == 1);
    CHECK(z1[0].pixels == 4);
    CHECK(z1[0].y0 == 1);
    CHECK(z1[0].y1 == 3);

    SECTION("YOLO gives one normalised line per box, classes from 0") {
        const std::string text = yoloText(slices, 1, labels.y(), labels.x());
        CHECK(std::count(text.begin(), text.end(), '\n') == 1);
        CHECK(text.rfind("0 ", 0) == 0);
        // centre of a box spanning x 1..3 in a 10 wide plane is 0.2, width 0.2
        CHECK(text.find("0.200000 0.250000 0.200000 0.250000") != std::string::npos);
        CHECK(yoloText(slices, 5, labels.y(), labels.x()).empty());
        CHECK(yoloText(slices, 1, 0, 0).empty());
    }
}

TEST_CASE("An export writes a dataset folder that accumulates", "[app][training]") {
    TempDir tmp;
    LabelVolume labels = twoCubes(2);
    const Array5 array = image(labels);

    TrainingExportOptions o;
    o.directory = tmp.dir.string();
    o.sample = "cells 01";
    o.minVoxels = 8;   // the dust is not training data
    o.provenance = nlohmann::json{{"step", "classic"}};
    const TrainingExportResult r = exportTrainingData(array, meta(), labels, o);

    CHECK(r.objects == 4);   // two objects in each of two frames
    CHECK(r.classes == 1);
    CHECK(r.frames == 2);
    CHECK(r.bytes > 0);
    CHECK(std::filesystem::exists(r.directory / "image.tif"));
    CHECK(std::filesystem::exists(r.directory / "instances.tif"));
    CHECK(std::filesystem::exists(r.directory / "semantic.tif"));
    CHECK(std::filesystem::exists(r.directory / "boxes.json"));
    CHECK(std::filesystem::exists(r.directory / "sample.json"));
    CHECK(std::filesystem::exists(tmp.dir / "classes.txt"));
    CHECK(std::filesystem::exists(tmp.dir / "index.jsonl"));

    SECTION("the instance masks are the labels, not a rendering of them") {
        const ImageStack<std::uint32_t> stack = readTiffStack<std::uint32_t>((r.directory / "instances.tif").string());
        REQUIRE(stack.dimension(0) == labels.t() * labels.z());
        REQUIRE(stack.dimension(1) == labels.y());
        REQUIRE(stack.dimension(2) == labels.x());
        CHECK(stack(1, 1, 1) == 1u);   // t0, z1, inside the first cube
        CHECK(stack(1, 0, 0) == 0u);
        CHECK(stack(9, 5, 6) == 2u);   // t1, z3, inside the second cube
    }

    SECTION("the boxes carry the geometry and the filter that made them") {
        std::ifstream in(r.directory / "boxes.json");
        const nlohmann::json j = nlohmann::json::parse(in);
        CHECK(j.at("min_voxels") == 8);
        CHECK(j.at("shape_tzyx") == nlohmann::json::array({2, 6, 8, 10}));
        CHECK(j.at("classes") == nlohmann::json::array({"object"}));
        REQUIRE(j.at("frames").size() == 2);
        const nlohmann::json& objects = j.at("frames")[0].at("objects");
        REQUIRE(objects.size() == 2);
        CHECK(objects[0].at("label") == 1);
        CHECK(objects[0].at("voxels") == 8);
        CHECK(objects[0].at("bbox_zyx") == nlohmann::json::array({1, 3, 1, 3, 1, 3}));
        // the dust was dropped from the plane boxes too, not only from the volume ones
        for (const nlohmann::json& s : j.at("frames")[0].at("slices")) CHECK(s.at("label") != 3);
    }

    SECTION("a second export is a second sample, and the index grows") {
        const TrainingExportResult again = exportTrainingData(array, meta(), labels, o);
        CHECK(again.directory != r.directory);
        CHECK(std::filesystem::exists(again.directory / "boxes.json"));
        std::ifstream index(tmp.dir / "index.jsonl");
        int lines = 0;
        for (std::string line; std::getline(index, line);)
            if (!line.empty()) {
                const nlohmann::json j = nlohmann::json::parse(line);
                CHECK(j.at("objects") == 4);
                CHECK(j.at("provenance").at("step") == "classic");
                ++lines;
            }
        CHECK(lines == 2);
    }

    SECTION("the class table at the root keeps the names of every sample") {
        LabelVolume named = twoCubes();
        named.stats()[0].cls = "nucleus";
        TrainingExportOptions o2 = o;
        o2.sample = "nuclei";
        exportTrainingData(image(named), meta(), named, o2);
        std::ifstream in(tmp.dir / "classes.txt");
        std::vector<std::string> names;
        for (std::string line; std::getline(in, line);)
            if (!line.empty()) names.push_back(line);
        CHECK(std::find(names.begin(), names.end(), "object") != names.end());
        CHECK(std::find(names.begin(), names.end(), "nucleus") != names.end());
    }
}

TEST_CASE("Slice output pairs an image with its boxes", "[app][training]") {
    TempDir tmp;
    LabelVolume labels = twoCubes();
    TrainingExportOptions o;
    o.directory = tmp.dir.string();
    o.sample = "slices";
    o.image = false;
    o.instances = false;
    o.semantic = false;
    o.slices = true;
    const TrainingExportResult r = exportTrainingData(image(labels), meta(), labels, o);
    // one txt and one tif per plane, empty planes included: a detector needs
    // the negatives as much as the positives
    for (Index z = 0; z < labels.z(); ++z) {
        const std::string stem = "t0_z" + std::to_string(z);
        CHECK(std::filesystem::exists(r.directory / "slices" / (stem + ".txt")));
        CHECK(std::filesystem::exists(r.directory / "slices" / (stem + ".tif")));
    }
    std::ifstream empty(r.directory / "slices" / "t0_z5.txt");
    CHECK(std::string(std::istreambuf_iterator<char>(empty), std::istreambuf_iterator<char>()).empty());
}

TEST_CASE("An export says what is wrong before writing anything", "[app][training]") {
    TempDir tmp;
    LabelVolume empty;
    TrainingExportOptions o;
    o.directory = tmp.dir.string();
    CHECK(validateTrainingExport(o, empty) == "this step produced no labels; run a segmentation step first");

    LabelVolume labels = twoCubes();
    o.directory.clear();
    CHECK(validateTrainingExport(o, labels) == "choose a folder for the training data");

    o.directory = tmp.dir.string();
    o.image = o.instances = o.semantic = o.boxes = o.slices = false;
    CHECK(validateTrainingExport(o, labels) == "nothing selected to write");
    CHECK_THROWS_AS(exportTrainingData(Array5{}, meta(), labels, o), std::runtime_error);

    SECTION("an image on another grid than the labels") {
        // two frames of labels over a one-frame image: the slices of t = 1
        // used to take the image of t = 0, and a z / y / x mismatch skipped
        // the slice images while image.tif was written all the same
        LabelVolume two = twoCubes(2);
        o.slices = true;
        o.sample = "mismatch";
        const Array5 one = image(twoCubes(1));
        CHECK(validateTrainingExport(o, two, one.dims()).find("not on the same grid") != std::string::npos);
        CHECK(validateTrainingExport(o, two, image(two).dims()).empty());
        CHECK_THROWS_AS(exportTrainingData(one, meta(), two, o), std::runtime_error);
        CHECK_FALSE(std::filesystem::exists(tmp.dir / "mismatch"));
        o.slices = false;
        o.boxes = true;   // no image written: the grid of an image is nobody's business
        CHECK(validateTrainingExport(o, two, one.dims()).empty());
    }
}

TEST_CASE("Every sample's class ids are positions in the dataset's class table", "[app][training]") {
    TempDir tmp;
    TrainingExportOptions o;
    o.directory = tmp.dir.string();
    o.image = false;
    o.instances = false;
    o.slices = true;

    LabelVolume a = twoCubes();
    for (LabelStats& s : a.stats()) s.cls = "nucleus";
    o.sample = "a";
    exportTrainingData(Array5{}, meta(), a, o);

    LabelVolume b = twoCubes();
    for (LabelStats& s : b.stats()) s.cls = s.id == 1 ? "cell" : "nucleus";
    o.sample = "b";
    const TrainingExportResult r = exportTrainingData(Array5{}, meta(), b, o);
    CHECK(r.classes == 2);

    std::vector<std::string> names;
    {
        std::ifstream in(tmp.dir / "classes.txt");
        for (std::string line; std::getline(in, line);)
            if (!line.empty()) names.push_back(line);
    }
    REQUIRE(names == std::vector<std::string>{"nucleus", "cell"});

    // b's own sorted table would say cell = 1, nucleus = 2; the dataset says
    // nucleus = 1, cell = 2, and that is what b has to write
    std::ifstream yolo(r.directory / "slices" / "t0_z1.txt");   // the plane of cube 1, the cell
    std::string line;
    REQUIRE(std::getline(yolo, line));
    CHECK(line.rfind("1 ", 0) == 0);   // YOLO counts from 0: cell is 1

    std::ifstream in(r.directory / "boxes.json");
    const nlohmann::json j = nlohmann::json::parse(in);
    CHECK(j.at("classes") == nlohmann::json::array({"nucleus", "cell"}));
    for (const nlohmann::json& object : j.at("frames")[0].at("objects")) {
        INFO(object.dump());
        CHECK(object.at("class_id") == (object.at("class") == "cell" ? 2 : 1));
    }

    const ImageStack<std::uint8_t> semantic = readTiffStack<std::uint8_t>((r.directory / "semantic.tif").string());
    CHECK(semantic(1, 1, 1) == 2);   // inside the cell
    CHECK(semantic(3, 5, 6) == 1);   // inside the nucleus
}

TEST_CASE("Classes are read from each object's own frame", "[app][training]") {
    // object 2 exists only in frame 1; the statistics table is on frame 0
    LabelVolume labels(2, 1, 16, 16);
    labels.paint(0, 0, 3, 3, 1.0, 0, 1);
    labels.paint(1, 0, 3, 3, 1.0, 0, 1);
    labels.paint(1, 0, 10, 10, 1.0, 0, 2);
    for (Index t = 0; t < 2; ++t) {
        labels.recomputeStats(t);
        for (LabelStats& s : labels.stats()) s.cls = "nucleus";   // as a segmentation step names them
    }
    labels.recomputeStats(0);   // the viewer is on frame 0
    const ClassTable classes = classTable(labels);
    REQUIRE(classes.names == std::vector<std::string>{"nucleus"});
    const Buffer<std::uint8_t> semantic = semanticVolume(labels, 1, classes);
    CHECK(semantic.data()[10 * 16 + 10] == 1);   // was 0: background
    CHECK(semantic.data()[3 * 16 + 3] == 1);
    const std::vector<LabelBox> boxes = boundingBoxes(labels, 1, classes, 1);
    REQUIRE(boxes.size() == 2);
    CHECK(boxes[1].label == 2);
    CHECK(boxes[1].className == "nucleus");
    CHECK(boxes[1].classId == 1);
    for (const SliceBox& s : sliceBoxes(labels, 1, classes, 1)) CHECK(s.classId == 1);
}
