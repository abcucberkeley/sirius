#include "core/training_export.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <sirius/tiff_io.hpp>

#include "core/cancel.hpp"

namespace sirius::app {

    namespace {

        const std::string kBackground = "background";

        std::string sanitised(std::string name) {
            for (char& c : name)
                if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|' || c == '\n')
                    c = '_';
            while (!name.empty() && (name.back() == ' ' || name.back() == '.')) name.pop_back();
            return name.empty() ? std::string("sample") : name;
        }

        std::string paddedIndex(Index i, int width) {
            std::string s = std::to_string(i);
            while (static_cast<int>(s.size()) < width) s.insert(s.begin(), '0');
            return s;
        }

        // (t * z, y, x) copy of every label plane, for the TIFF writer.
        Buffer<std::uint32_t> instanceStack(const LabelVolume& labels) {
            Buffer<std::uint32_t> out(Shape{labels.t() * labels.z(), labels.y(), labels.x()});
            const std::size_t planeSize = static_cast<std::size_t>(labels.y() * labels.x());
            for (Index t = 0; t < labels.t(); ++t)
                for (Index z = 0; z < labels.z(); ++z)
                    std::memcpy(out.data() + (static_cast<std::size_t>(t * labels.z() + z)) * planeSize, labels.plane(t, z),
                                planeSize * sizeof(std::uint32_t));
            return out;
        }

    } // namespace

    int ClassTable::idOf(const std::string& name) const noexcept {
        for (std::size_t i = 0; i < names.size(); ++i)
            if (names[i] == name) return static_cast<int>(i) + 1;
        return 0;
    }

    const std::string& ClassTable::nameOf(int id) const noexcept {
        if (id <= 0 || static_cast<std::size_t>(id) > names.size()) return kBackground;
        return names[static_cast<std::size_t>(id) - 1];
    }

    namespace {
        // The class of object `id` in frame t: what its own frame says (the
        // statistics table describes one frame, which need not be t), and
        // "object" for an object nothing was said about.
        std::string classOf(const LabelVolume& labels, Index t, std::uint32_t id) {
            std::string cls = labels.annotationOf(t, id).cls;
            return cls.empty() ? std::string("object") : cls;
        }
    } // namespace

    ClassTable classTable(const LabelVolume& labels) {
        ClassTable table;
        if (labels.empty()) return table;
        // Every object of every frame, not the rows of the table on screen:
        // an object absent from that frame used to be left out, and then
        // written as background in the semantic mask. Labels nothing was
        // said about land in "object".
        std::vector<std::string> seen;
        std::unordered_set<std::uint32_t> ids;
        for (Index t = 0; t < labels.t(); ++t) {
            ids.clear();
            const std::uint32_t* v = labels.volume(t);
            std::uint32_t last = 0;
            for (Index i = 0, n = labels.volumeSize(); i < n; ++i)
                if (v[i] != 0 && v[i] != last) {
                    last = v[i];
                    ids.insert(last);
                }
            for (std::uint32_t id : ids) {
                std::string cls = classOf(labels, t, id);
                if (std::find(seen.begin(), seen.end(), cls) == seen.end()) seen.push_back(std::move(cls));
            }
        }
        std::sort(seen.begin(), seen.end());
        table.names = std::move(seen);
        return table;
    }

    std::vector<LabelBox> boundingBoxes(const LabelVolume& labels, Index t, const ClassTable& classes, std::uint64_t minVoxels) {
        std::vector<LabelBox> out;
        if (labels.empty() || t < 0 || t >= labels.t()) return out;
        const Index nz = labels.z(), ny = labels.y(), nx = labels.x();
        const std::uint32_t* v = labels.volume(t);

        struct Acc {
            Index z0 = std::numeric_limits<Index>::max(), z1 = 0, y0 = std::numeric_limits<Index>::max(), y1 = 0;
            Index x0 = std::numeric_limits<Index>::max(), x1 = 0;
            std::uint64_t voxels = 0;
            double sz = 0.0, sy = 0.0, sx = 0.0;
        };
        std::map<std::uint32_t, Acc> acc;   // ordered: the result is sorted by id
        for (Index z = 0; z < nz; ++z)
            for (Index y = 0; y < ny; ++y) {
                const std::uint32_t* row = v + (z * ny + y) * nx;
                for (Index x = 0; x < nx; ++x) {
                    const std::uint32_t id = row[x];
                    if (id == 0) continue;
                    Acc& a = acc[id];
                    a.z0 = std::min(a.z0, z);
                    a.z1 = std::max(a.z1, z + 1);
                    a.y0 = std::min(a.y0, y);
                    a.y1 = std::max(a.y1, y + 1);
                    a.x0 = std::min(a.x0, x);
                    a.x1 = std::max(a.x1, x + 1);
                    ++a.voxels;
                    a.sz += static_cast<double>(z);
                    a.sy += static_cast<double>(y);
                    a.sx += static_cast<double>(x);
                }
            }

        out.reserve(acc.size());
        for (const auto& [id, a] : acc) {
            if (a.voxels < std::max<std::uint64_t>(minVoxels, 1)) continue;
            LabelBox b;
            b.label = id;
            b.t = t;
            b.bbox = {a.z0, a.z1, a.y0, a.y1, a.x0, a.x1};
            b.voxels = a.voxels;
            const double n = static_cast<double>(a.voxels);
            b.cz = a.sz / n;
            b.cy = a.sy / n;
            b.cx = a.sx / n;
            b.touchesBorder = a.z0 == 0 || a.z1 == nz || a.y0 == 0 || a.y1 == ny || a.x0 == 0 || a.x1 == nx;
            b.className = classOf(labels, t, id);
            b.classId = classes.idOf(b.className);
            out.push_back(std::move(b));
        }
        return out;
    }

    std::vector<SliceBox> sliceBoxes(const LabelVolume& labels, Index t, const ClassTable& classes, std::uint64_t minPixels) {
        std::vector<SliceBox> out;
        if (labels.empty() || t < 0 || t >= labels.t()) return out;
        const Index nz = labels.z(), ny = labels.y(), nx = labels.x();
        const std::uint32_t* v = labels.volume(t);

        struct Acc {
            Index y0 = std::numeric_limits<Index>::max(), y1 = 0, x0 = std::numeric_limits<Index>::max(), x1 = 0;
            std::uint64_t pixels = 0;
        };
        std::unordered_map<std::uint32_t, int> idOf;   // class id per label, looked up once
        for (Index z = 0; z < nz; ++z) {
            std::map<std::uint32_t, Acc> acc;
            for (Index y = 0; y < ny; ++y) {
                const std::uint32_t* row = v + (z * ny + y) * nx;
                for (Index x = 0; x < nx; ++x) {
                    const std::uint32_t id = row[x];
                    if (id == 0) continue;
                    Acc& a = acc[id];
                    a.y0 = std::min(a.y0, y);
                    a.y1 = std::max(a.y1, y + 1);
                    a.x0 = std::min(a.x0, x);
                    a.x1 = std::max(a.x1, x + 1);
                    ++a.pixels;
                }
            }
            for (const auto& [id, a] : acc) {
                if (a.pixels < std::max<std::uint64_t>(minPixels, 1)) continue;
                auto it = idOf.find(id);
                if (it == idOf.end()) it = idOf.emplace(id, classes.idOf(classOf(labels, t, id))).first;
                out.push_back(SliceBox{z, id, it->second, a.y0, a.y1, a.x0, a.x1, a.pixels});
            }
        }
        return out;
    }

    Buffer<std::uint8_t> semanticVolume(const LabelVolume& labels, Index t, const ClassTable& classes) {
        Buffer<std::uint8_t> out(Shape{labels.z(), labels.y(), labels.x()});
        std::memset(out.data(), 0, out.bytes());
        if (labels.empty() || t < 0 || t >= labels.t()) return out;
        // one lookup per label, not per voxel
        std::unordered_map<std::uint32_t, std::uint8_t> idOf;
        const std::uint32_t* v = labels.volume(t);
        const Index n = labels.volumeSize();
        std::uint8_t* dst = out.data();
        for (Index i = 0; i < n; ++i) {
            const std::uint32_t id = v[i];
            if (id == 0) continue;
            auto it = idOf.find(id);
            if (it == idOf.end()) {
                const int c = classes.idOf(classOf(labels, t, id));
                it = idOf.emplace(id, static_cast<std::uint8_t>(std::clamp(c, 0, 255))).first;
            }
            dst[i] = it->second;
        }
        return out;
    }

    std::string yoloText(const std::vector<SliceBox>& boxes, Index z, Index height, Index width) {
        std::string out;
        if (height <= 0 || width <= 0) return out;
        const double h = static_cast<double>(height), w = static_cast<double>(width);
        char line[192];
        for (const SliceBox& b : boxes) {
            if (b.z != z) continue;
            const double cx = (static_cast<double>(b.x0 + b.x1) * 0.5) / w;
            const double cy = (static_cast<double>(b.y0 + b.y1) * 0.5) / h;
            const double bw = static_cast<double>(b.x1 - b.x0) / w;
            const double bh = static_cast<double>(b.y1 - b.y0) / h;
            // YOLO counts classes from 0; the semantic mask keeps 0 for background
            std::snprintf(line, sizeof line, "%d %.6f %.6f %.6f %.6f\n", std::max(0, b.classId - 1), cx, cy, bw, bh);
            out += line;
        }
        return out;
    }

    std::string validateTrainingExport(const TrainingExportOptions& o, const LabelVolume& labels) {
        if (o.directory.empty()) return "choose a folder for the training data";
        if (labels.empty()) return "this step produced no labels; run a segmentation step first";
        if (!o.image && !o.instances && !o.semantic && !o.boxes && !o.slices) return "nothing selected to write";
        return {};
    }

    std::string validateTrainingExport(const TrainingExportOptions& o, const LabelVolume& labels, const Dims5& imageDims) {
        if (std::string problem = validateTrainingExport(o, labels); !problem.empty()) return problem;
        if (!(o.image || o.slices)) return {};
        if (imageDims.t != labels.t() || imageDims.z != labels.z() || imageDims.y != labels.y() || imageDims.x != labels.x()) {
            auto shape = [](Index t, Index z, Index y, Index x) {
                return "t" + std::to_string(t) + " z" + std::to_string(z) + " y" + std::to_string(y) + " x" + std::to_string(x);
            };
            return "the image (" + shape(imageDims.t, imageDims.z, imageDims.y, imageDims.x) + ") and the labels (" +
                   shape(labels.t(), labels.z(), labels.y(), labels.x()) +
                   ") are not on the same grid; export the image from the step the labels belong to";
        }
        return {};
    }

    TrainingExportResult exportTrainingData(const Array5& array, const DatasetMeta& meta, const LabelVolume& labels,
                                            const TrainingExportOptions& options,
                                            const std::function<void(double, const std::string&)>& progress,
                                            const std::function<bool()>& cancelled) {
        // An empty array writes no image (a caller that wants none passes
        // none); one that is there has to fit the labels.
        const std::string problem =
            array.empty() ? validateTrainingExport(options, labels) : validateTrainingExport(options, labels, array.dims());
        if (!problem.empty()) throw std::runtime_error(problem);
        auto report = [&](double f, const std::string& what) {
            if (progress) progress(std::clamp(f, 0.0, 1.0), what);
        };
        auto checkCancel = [&] {
            if (cancelled && cancelled()) throw CancelledError();
        };

        const std::filesystem::path root(options.directory);
        std::filesystem::create_directories(root);
        std::filesystem::path dir = root / sanitised(options.sample);
        if (!options.overwrite)
            for (int n = 2; std::filesystem::exists(dir); ++n) dir = root / (sanitised(options.sample) + "-" + std::to_string(n));
        std::filesystem::create_directories(dir);

        TrainingExportResult result;
        result.directory = dir;
        result.frames = labels.t();
        auto note = [&](const std::filesystem::path& p) {
            result.files.push_back(std::filesystem::relative(p, root).generic_string());
            std::error_code ec;
            // file_size answers uintmax_t(-1) on failure, which added to a
            // running total reports sixteen exabytes written
            const std::uintmax_t size = std::filesystem::file_size(p, ec);
            if (!ec) result.bytes += size;
        };

        const ClassTable own = classTable(labels);
        result.classes = own.size();
        const std::string sampleName = dir.filename().string();

        // The class table belongs to the dataset, not to one sample: written
        // (and grown) at the root so every sample agrees on the ids -- and
        // the ids this sample writes (semantic mask, boxes, YOLO) are
        // positions in it. A sample's own sorted table numbered "cell" 1 in
        // a dataset whose classes.txt already said "nucleus, cell".
        ClassTable classes;
        {
            std::vector<std::string>& known = classes.names;
            if (std::ifstream in(root / "classes.txt"); in) {
                for (std::string line; std::getline(in, line);)
                    if (!line.empty()) known.push_back(line);
            }
            bool grew = false;
            for (const std::string& name : own.names)
                if (std::find(known.begin(), known.end(), name) == known.end()) {
                    known.push_back(name);
                    grew = true;
                }
            if (grew || !std::filesystem::exists(root / "classes.txt")) {
                std::ofstream out(root / "classes.txt", std::ios::trunc);
                if (!out) throw std::runtime_error("cannot write " + (root / "classes.txt").string());
                for (const std::string& name : known) out << name << "\n";
            }
        }

        TiffWriteOptions tw;
        tw.compression = TiffCompression::Deflate;
        tw.compressionLevel = 6;
        tw.predictor = true;
        tw.bigTiff = true;
        tw.xPixelUm = meta.voxelUm[0];
        tw.yPixelUm = meta.voxelUm[1];

        if (options.image && !array.empty()) {
            checkCancel();
            report(0.05, "writing the image");
            ExportOptions eo;
            eo.path = (dir / "image.tif").string();
            eo.format = ExportFormat::Tiff;
            eo.dtype = options.imageDtype;
            eo.scaling = options.scaling;
            eo.rangeLo = options.rangeLo;
            eo.rangeHi = options.rangeHi;
            eo.percentileLo = options.percentileLo;
            eo.percentileHi = options.percentileHi;
            eo.tiff.compression = TiffCompression::Deflate;
            eo.tiff.predictor = true;
            exportArray(array, meta, nullptr, eo, [&](double f, const std::string& what) { report(0.05 + 0.35 * f, what); }, cancelled);
            note(dir / "image.tif");
        }

        if (options.instances) {
            checkCancel();
            report(0.45, "writing the instance masks");
            const Buffer<std::uint32_t> stack = instanceStack(labels);
            tw.description = "SIRIUS instance labels · order tzyx · t" + std::to_string(labels.t()) + " z" + std::to_string(labels.z());
            writeTiffStack<std::uint32_t>((dir / "instances.tif").string(), stack.view(), tw);
            note(dir / "instances.tif");
        }

        if (options.semantic) {
            checkCancel();
            report(0.6, "writing the semantic masks");
            Buffer<std::uint8_t> stack(Shape{labels.t() * labels.z(), labels.y(), labels.x()});
            const std::size_t volumeBytes = static_cast<std::size_t>(labels.volumeSize());
            for (Index t = 0; t < labels.t(); ++t) {
                const Buffer<std::uint8_t> one = semanticVolume(labels, t, classes);
                std::memcpy(stack.data() + static_cast<std::size_t>(t) * volumeBytes, one.data(), volumeBytes);
            }
            tw.description = "SIRIUS semantic mask · order tzyx · classes " + std::to_string(classes.size());
            writeTiffStack<std::uint8_t>((dir / "semantic.tif").string(), stack.view(), tw);
            note(dir / "semantic.tif");
        }

        std::vector<std::vector<SliceBox>> perFrameSlices;
        nlohmann::json boxesJson;
        if (options.boxes || options.slices) {
            checkCancel();
            report(0.72, "measuring the objects");
            nlohmann::json frames = nlohmann::json::array();
            for (Index t = 0; t < labels.t(); ++t) {
                checkCancel();
                const std::vector<LabelBox> boxes = boundingBoxes(labels, t, classes, options.minVoxels);
                std::vector<SliceBox> slices = sliceBoxes(labels, t, classes, 1);
                // a plane box of an object the volume filter dropped is not training data either
                if (options.minVoxels > 1) {
                    std::vector<std::uint32_t> keep;
                    keep.reserve(boxes.size());
                    for (const LabelBox& b : boxes) keep.push_back(b.label);
                    std::sort(keep.begin(), keep.end());
                    slices.erase(std::remove_if(slices.begin(), slices.end(),
                                                [&](const SliceBox& s) { return !std::binary_search(keep.begin(), keep.end(), s.label); }),
                                 slices.end());
                }
                result.objects += boxes.size();
                result.sliceObjects += slices.size();

                nlohmann::json objects = nlohmann::json::array();
                for (const LabelBox& b : boxes)
                    objects.push_back({{"label", b.label},
                                       {"class", b.className},
                                       {"class_id", b.classId},
                                       {"bbox_zyx", {b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], b.bbox[4], b.bbox[5]}},
                                       {"voxels", b.voxels},
                                       {"centroid_zyx", {b.cz, b.cy, b.cx}},
                                       {"touches_border", b.touchesBorder}});
                nlohmann::json planes = nlohmann::json::array();
                for (const SliceBox& s : slices)
                    planes.push_back({{"z", s.z},
                                      {"label", s.label},
                                      {"class_id", s.classId},
                                      {"bbox_yx", {s.y0, s.y1, s.x0, s.x1}},
                                      {"pixels", s.pixels}});
                frames.push_back({{"t", t}, {"objects", objects}, {"slices", planes}});
                perFrameSlices.push_back(std::move(slices));
            }
            boxesJson = {{"version", 1},
                         {"sample", sampleName},
                         {"shape_tzyx", {labels.t(), labels.z(), labels.y(), labels.x()}},
                         {"voxel_um", {meta.voxelUm[0], meta.voxelUm[1], meta.voxelUm[2]}},
                         {"classes", classes.names},
                         {"min_voxels", options.minVoxels},
                         {"frames", frames}};
        }

        if (options.boxes) {
            std::ofstream out(dir / "boxes.json", std::ios::trunc);
            if (!out) throw std::runtime_error("cannot write " + (dir / "boxes.json").string());
            out << boxesJson.dump(2) << "\n";
            out.close();
            note(dir / "boxes.json");
        }

        if (options.slices) {
            checkCancel();
            report(0.82, "writing the slices");
            const std::filesystem::path slicesDir = dir / "slices";
            std::filesystem::create_directories(slicesDir);
            const Index ny = labels.y(), nx = labels.x();
            const int zWidth = static_cast<int>(std::to_string(std::max<Index>(labels.z() - 1, 0)).size());
            const int tWidth = static_cast<int>(std::to_string(std::max<Index>(labels.t() - 1, 0)).size());
            const bool haveImage = !array.empty();   // on the labels' grid: validated above
            for (Index t = 0; t < labels.t(); ++t)
                for (Index z = 0; z < labels.z(); ++z) {
                    checkCancel();
                    const std::string stem = "t" + paddedIndex(t, tWidth) + "_z" + paddedIndex(z, zWidth);
                    std::ofstream txt(slicesDir / (stem + ".txt"), std::ios::trunc);
                    if (!txt) throw std::runtime_error("cannot write " + (slicesDir / (stem + ".txt")).string());
                    txt << yoloText(perFrameSlices[static_cast<std::size_t>(t)], z, ny, nx);
                    txt.close();
                    if (!haveImage) continue;
                    // 8 bit, scaled per plane: what a 2D detector reads
                    const float* src = array.plane(0, t, z);
                    float lo = std::numeric_limits<float>::max(), hi = std::numeric_limits<float>::lowest();
                    for (Index i = 0, n = ny * nx; i < n; ++i) {
                        lo = std::min(lo, src[i]);
                        hi = std::max(hi, src[i]);
                    }
                    const float span = hi > lo ? hi - lo : 1.0f;
                    Buffer<std::uint8_t> plane(Shape{1, ny, nx});
                    for (Index i = 0, n = ny * nx; i < n; ++i)
                        plane.data()[i] = static_cast<std::uint8_t>(std::clamp((src[i] - lo) / span * 255.0f + 0.5f, 0.0f, 255.0f));
                    TiffWriteOptions pw;
                    pw.compression = TiffCompression::Deflate;
                    pw.bigTiff = false;
                    writeTiffStack<std::uint8_t>((slicesDir / (stem + ".tif")).string(), plane.view(), pw);
                }
            result.files.push_back(std::filesystem::relative(slicesDir, root).generic_string() + "/");
        }

        if (!options.pipelineToml.empty()) {
            std::ofstream out(dir / "pipeline.sirius.toml", std::ios::trunc);
            if (!out) throw std::runtime_error("cannot write " + (dir / "pipeline.sirius.toml").string());
            out << options.pipelineToml;
            out.close();
            note(dir / "pipeline.sirius.toml");
        }

        // Per sample record, and one line appended to the dataset index: a
        // loader reads the index, not the folder listing.
        nlohmann::json record{{"version", 1},
                              {"sample", sampleName},
                              {"shape_tzyx", {labels.t(), labels.z(), labels.y(), labels.x()}},
                              {"channels", array.empty() ? 0 : array.dims().c},
                              {"voxel_um", {meta.voxelUm[0], meta.voxelUm[1], meta.voxelUm[2]}},
                              {"classes", classes.names},
                              {"objects", result.objects},
                              {"slice_objects", result.sliceObjects},
                              {"files", result.files},
                              {"source", meta.sourcePath}};
        if (!options.provenance.is_null()) record["provenance"] = options.provenance;
        {
            std::ofstream out(dir / "sample.json", std::ios::trunc);
            if (!out) throw std::runtime_error("cannot write " + (dir / "sample.json").string());
            out << record.dump(2) << "\n";
        }
        {
            std::ofstream index(root / "index.jsonl", std::ios::app);
            if (!index) throw std::runtime_error("cannot write " + (root / "index.jsonl").string());
            index << record.dump() << "\n";
        }
        report(1.0, "done");
        return result;
    }

} // namespace sirius::app
