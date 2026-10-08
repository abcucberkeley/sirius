#include "core/serialize.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "core/errors.hpp"

namespace sirius::app {

    using json = nlohmann::json;

    namespace {
        template <typename T> void read(const json& j, const char* key, T& out) {
            const auto it = j.find(key);
            if (it == j.end() || it->is_null()) return;
            try {
                out = it->get<T>();
            } catch (const json::exception& e) {
                throw std::runtime_error(std::string("'") + key + "': " + e.what());
            }
        }

        json color(const std::array<float, 3>& c) { return json::array({c[0], c[1], c[2]}); }
        std::array<float, 3> colorFrom(const json& j, std::array<float, 3> fallback) {
            if (!j.is_array() || j.size() != 3) return fallback;
            return {j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
        }
    } // namespace

    // --- DatasetMeta ---------------------------------------------------------------------

    const char* dtypeName(PixelType t) noexcept {
        switch (t) {
            case PixelType::UInt8: return "uint8";
            case PixelType::Int8: return "int8";
            case PixelType::UInt16: return "uint16";
            case PixelType::Int16: return "int16";
            case PixelType::UInt32: return "uint32";
            case PixelType::Int32: return "int32";
            case PixelType::Float32: return "float32";
            case PixelType::Float64: return "float64";
        }
        return "float32";
    }

    PixelType pixelTypeFromName(const std::string& d) noexcept {
        if (d == "uint8" || d == "bool") return PixelType::UInt8;
        if (d == "int8") return PixelType::Int8;
        if (d == "uint16") return PixelType::UInt16;
        if (d == "int16") return PixelType::Int16;
        if (d == "uint32") return PixelType::UInt32;
        if (d == "int32") return PixelType::Int32;
        if (d == "float64") return PixelType::Float64;
        return PixelType::Float32;
    }

    json toJson(const DatasetMeta& m) {
        // not SIM: present alone, so no angle or phase count reads as a detection
        json sim = {{"present", m.sim.present}};
        if (m.sim.present) {
            sim["ndirs"] = m.sim.ndirs;
            sim["nphases"] = m.sim.nphases;
            sim["fast_si"] = m.sim.fastSi;
            if (!m.sim.isShorthand()) sim["layout"] = m.sim.storage;
        }
        json channels = json::array();
        for (const ChannelInfo& c : m.channels)
            channels.push_back({{"label", c.label}, {"wavelength_nm", c.wavelengthNm}, {"color", color(c.color)}, {"exposure", c.exposure}});
        json tiles = json::array();
        for (const TileInfo& t : m.tiles)
            tiles.push_back({{"name", t.name},
                             {"position_um", {t.positionUm[0], t.positionUm[1], t.positionUm[2]}},
                             {"grid_index", {t.gridIndex[0], t.gridIndex[1], t.gridIndex[2]}}});
        return {{"name", m.name},
                {"source_path", m.sourcePath},
                {"format", m.format},
                {"dims", {m.dims.c, m.dims.t, m.dims.z, m.dims.y, m.dims.x}},
                {"dtype", dtypeName(m.sourceType)},
                {"bytes", m.bytesOnDisk},
                {"voxel_um", {m.voxelUm[0], m.voxelUm[1], m.voxelUm[2]}},
                {"frame_interval_s", m.frameIntervalS},
                {"channels", channels},
                {"acquisition", m.acquisition},
                {"sim", sim},
                {"rgb", m.rgb},
                {"light_sheet", m.lightSheet},
                {"sheet_angle_deg", m.sheetAngleDeg},
                {"tiles", tiles},
                {"tile_index", m.tileIndex}};
    }

    DatasetMeta datasetMetaFromJson(const json& j) {
        if (!j.is_object()) throw std::runtime_error("dataset meta: not an object");
        DatasetMeta m;
        read(j, "name", m.name);
        read(j, "source_path", m.sourcePath);
        read(j, "format", m.format);
        if (auto it = j.find("dims"); it != j.end()) {
            if (!it->is_array() || it->size() != 5) throw std::runtime_error("dataset meta: dims is not (c, t, z, y, x)");
            m.dims = Dims5{(*it)[0].get<Index>(), (*it)[1].get<Index>(), (*it)[2].get<Index>(), (*it)[3].get<Index>(), (*it)[4].get<Index>()};
        }
        if (auto it = j.find("dtype"); it != j.end() && it->is_string()) m.sourceType = pixelTypeFromName(it->get<std::string>());
        read(j, "bytes", m.bytesOnDisk);
        if (auto it = j.find("voxel_um"); it != j.end() && it->is_array() && it->size() == 3)
            for (std::size_t k = 0; k < 3; ++k) m.voxelUm[k] = (*it)[k].get<double>();
        read(j, "frame_interval_s", m.frameIntervalS);
        if (auto it = j.find("channels"); it != j.end() && it->is_array())
            for (const json& c : *it) {
                ChannelInfo ci;
                read(c, "label", ci.label);
                read(c, "wavelength_nm", ci.wavelengthNm);
                if (c.contains("color")) ci.color = colorFrom(c["color"], ci.color);
                read(c, "exposure", ci.exposure);
                m.channels.push_back(std::move(ci));
            }
        read(j, "acquisition", m.acquisition);
        if (auto it = j.find("sim"); it != j.end() && it->is_object()) {
            read(*it, "present", m.sim.present);
            read(*it, "ndirs", m.sim.ndirs);
            read(*it, "nphases", m.sim.nphases);
            read(*it, "fast_si", m.sim.fastSi);
            std::string layout;
            read(*it, "layout", layout);
            if (!layout.empty()) {
                const SimLayout general = SimLayout::fromText(layout);
                m.sim.storage = general.storage;
                m.sim.ndirs = general.ndirs;
                m.sim.nphases = general.nphases;
                m.sim.fastSi = general.fastSi;
            }
        }
        read(j, "rgb", m.rgb);
        read(j, "light_sheet", m.lightSheet);
        read(j, "sheet_angle_deg", m.sheetAngleDeg);
        if (auto it = j.find("tiles"); it != j.end() && it->is_array())
            for (const json& t : *it) {
                TileInfo ti;
                read(t, "name", ti.name);
                if (t.contains("position_um") && t["position_um"].is_array() && t["position_um"].size() == 3)
                    for (std::size_t k = 0; k < 3; ++k) ti.positionUm[k] = t["position_um"][k].get<double>();
                if (t.contains("grid_index") && t["grid_index"].is_array() && t["grid_index"].size() == 3)
                    for (std::size_t k = 0; k < 3; ++k) ti.gridIndex[k] = t["grid_index"][k].get<Index>();
                m.tiles.push_back(std::move(ti));
            }
        read(j, "tile_index", m.tileIndex);
        return m;
    }

    // --- StepReport ----------------------------------------------------------------------

    std::optional<StepReport::State> stepStateFromString(const std::string& s) noexcept {
        for (StepReport::State st : {StepReport::State::Running, StepReport::State::Ran, StepReport::State::Cached, StepReport::State::Skipped,
                                     StepReport::State::Failed})
            if (s == toString(st)) return st;
        return std::nullopt;
    }

    json toJson(const StepReport& r) {
        return {{"id", r.id}, {"index", r.index}, {"state", toString(r.state)}, {"seconds", r.seconds}, {"note", r.note}, {"error", r.error}};
    }

    StepReport stepReportFromJson(const json& j) {
        if (!j.is_object()) throw std::runtime_error("step report: not an object");
        StepReport r;
        read(j, "id", r.id);
        read(j, "index", r.index);
        read(j, "seconds", r.seconds);
        read(j, "note", r.note);
        read(j, "error", r.error);
        std::string state;
        read(j, "state", state);
        const auto st = stepStateFromString(state);
        if (!st) throw std::runtime_error("step report: unknown state '" + state + "'");
        r.state = *st;
        return r;
    }

    // --- Lineage -------------------------------------------------------------------------

    json lineageToJson(const Lineage& lineage) {
        json j = json::object();
        for (const auto& [child, parent] : lineage) j[std::to_string(child)] = parent;
        return j;
    }

    // --- Diagnostics ---------------------------------------------------------------------

    namespace {
        const char* kindName(DiagnosticsKind k) {
            switch (k) {
                case DiagnosticsKind::Generic: return "generic";
                case DiagnosticsKind::Sim: return "sim";
                case DiagnosticsKind::Deconvolve: return "deconvolve";
                case DiagnosticsKind::Contrast: return "contrast";
                case DiagnosticsKind::Segment: return "segment";
                case DiagnosticsKind::Alignment: return "alignment";
                case DiagnosticsKind::Volume: return "volume";
            }
            return "generic";
        }
        DiagnosticsKind kindFrom(const std::string& s) {
            for (DiagnosticsKind k : {DiagnosticsKind::Generic, DiagnosticsKind::Sim, DiagnosticsKind::Deconvolve, DiagnosticsKind::Contrast,
                                      DiagnosticsKind::Segment, DiagnosticsKind::Alignment, DiagnosticsKind::Volume})
                if (s == kindName(k)) return k;
            return DiagnosticsKind::Generic;
        }
        const char* markName(DiagnosticMark::Kind k) {
            return k == DiagnosticMark::Kind::Circle ? "circle" : k == DiagnosticMark::Kind::Ring ? "ring"
                                                                                                  : "cross";
        }
        DiagnosticMark::Kind markFrom(const std::string& s) {
            return s == "circle" ? DiagnosticMark::Kind::Circle : s == "ring" ? DiagnosticMark::Kind::Ring
                                                                              : DiagnosticMark::Kind::Cross;
        }
        json facts(const std::vector<DiagnosticFact>& fs) {
            json a = json::array();
            for (const DiagnosticFact& f : fs) a.push_back({f.key, f.value});
            return a;
        }
        std::vector<DiagnosticFact> factsFrom(const json& j) {
            std::vector<DiagnosticFact> out;
            if (j.is_array())
                for (const json& f : j)
                    if (f.is_array() && f.size() == 2) out.push_back({f[0].get<std::string>(), f[1].get<std::string>()});
            return out;
        }
        // NaN and infinity are not JSON; nlohmann writes them as null
        json number(double v) { return std::isfinite(v) ? json(v) : json(nullptr); }
        double numberFrom(const json& j) { return j.is_number() ? j.get<double>() : std::numeric_limits<double>::quiet_NaN(); }
        json numbers(const std::vector<double>& v) {
            json a = json::array();
            for (double x : v) a.push_back(number(x));
            return a;
        }
        std::vector<double> numbersFrom(const json& j) {
            std::vector<double> out;
            if (j.is_array())
                for (const json& x : j) out.push_back(numberFrom(x));
            return out;
        }
    } // namespace

    EncodedDiagnostics encodeDiagnostics(const Diagnostics& d, const std::string& prefix) {
        EncodedDiagnostics out;
        json images = json::array();
        for (std::size_t i = 0; i < d.images.size(); ++i) {
            const DiagnosticImage& im = d.images[i];
            if (static_cast<Index>(im.values.size()) != im.rows * im.cols)
                throw std::invalid_argument("diagnostics: image '" + im.title + "' holds " + std::to_string(im.values.size()) + " values, not " +
                                            std::to_string(im.rows) + " x " + std::to_string(im.cols));
            const std::string name = prefix + "img" + std::to_string(i);
            json marks = json::array();
            for (const DiagnosticMark& m : im.marks)
                marks.push_back({{"kind", markName(m.kind)}, {"x", m.x}, {"y", m.y}, {"radius", m.radius}, {"accent", m.accent}, {"text", m.text}});
            images.push_back({{"title", im.title}, {"meta", im.meta}, {"rows", im.rows}, {"cols", im.cols}, {"log", im.logScale}, {"marks", marks}, {"tensor", name}});
            rpc::Tensor t;
            t.name = name;
            t.dtype = "float32";
            t.shape = {im.rows, im.cols};
            t.bytes.resize(im.values.size() * sizeof(float));
            if (!im.values.empty()) std::memcpy(t.bytes.data(), im.values.data(), t.bytes.size());
            out.tensors.push_back(std::move(t));
        }
        json tabs = json::array();
        for (const DiagnosticTab& t : d.tabs) tabs.push_back({{"name", t.name}, {"images", t.images}});
        json curves = json::array();
        for (const DiagnosticCurve& c : d.curves)
            curves.push_back({{"title", c.title},
                              {"x", numbers(c.x)},
                              {"y", numbers(c.y)},
                              {"stop_x", c.stopX ? number(*c.stopX) : json(nullptr)},
                              {"labels", {c.leftLabel, c.midLabel, c.rightLabel}},
                              {"log_y", c.logY}});
        json histograms = json::array();
        for (const DiagnosticHistogram& h : d.histograms)
            histograms.push_back({{"channel", h.channel},
                                  {"color", color(h.color)},
                                  {"bins", numbers(h.bins)},
                                  {"bin_lo", number(h.binLo)},
                                  {"bin_hi", number(h.binHi)},
                                  {"lo", number(h.lo)},
                                  {"hi", number(h.hi)},
                                  {"gamma", number(h.gamma)}});
        json j = {{"kind", kindName(d.kind)}, {"summary", d.summary}, {"footer", d.footer}, {"images", images}, {"tabs", tabs}, {"curves", curves}, {"histograms", histograms}, {"facts", facts(d.facts)}, {"warnings", d.warnings}, {"table", nullptr}, {"alignment", nullptr}};
        if (d.table) {
            json accent = json::array();
            for (const auto& [r, c] : d.table->accentCells) accent.push_back({r, c});
            j["table"] = {{"caption", d.table->caption}, {"header", d.table->header}, {"rows", d.table->rows}, {"accent", accent}};
        }
        if (d.alignment)
            j["alignment"] = {{"grid", {d.alignment->gridRows, d.alignment->gridCols}},
                              {"tiles", d.alignment->tileNames},
                              {"highlighted", d.alignment->highlightedTile},
                              {"shift_stats", facts(d.alignment->shiftStats)}};
        out.json = std::move(j);
        return out;
    }

    Diagnostics decodeDiagnostics(const json& j, const std::vector<rpc::Tensor>& tensors) {
        if (!j.is_object()) throw ProtocolError("diagnostics: not an object");
        Diagnostics d;
        std::string kind;
        read(j, "kind", kind);
        d.kind = kindFrom(kind);
        read(j, "summary", d.summary);
        read(j, "footer", d.footer);
        if (j.contains("images") && j["images"].is_array())
            for (const json& im : j["images"]) {
                DiagnosticImage img;
                read(im, "title", img.title);
                read(im, "meta", img.meta);
                read(im, "rows", img.rows);
                read(im, "cols", img.cols);
                read(im, "log", img.logScale);
                if (im.contains("marks") && im["marks"].is_array())
                    for (const json& m : im["marks"]) {
                        DiagnosticMark mk;
                        std::string mkind;
                        read(m, "kind", mkind);
                        mk.kind = markFrom(mkind);
                        read(m, "x", mk.x);
                        read(m, "y", mk.y);
                        read(m, "radius", mk.radius);
                        read(m, "accent", mk.accent);
                        read(m, "text", mk.text);
                        img.marks.push_back(std::move(mk));
                    }
                const std::string name = im.value("tensor", std::string());
                const rpc::Tensor* t = nullptr;
                for (const rpc::Tensor& c : tensors)
                    if (c.name == name) t = &c;
                if (!t) throw ProtocolError("diagnostics: the values of image '" + img.title + "' (tensor '" + name + "') are missing");
                if (img.rows < 0 || img.cols < 0 || t->numel() != img.rows * img.cols)
                    throw ProtocolError("diagnostics: tensor '" + name + "' does not hold " + std::to_string(img.rows) + " x " + std::to_string(img.cols) + " values");
                const float* p = t->asFloat32();
                img.values.assign(p, p + t->numel());
                d.images.push_back(std::move(img));
            }
        if (j.contains("tabs") && j["tabs"].is_array())
            for (const json& t : j["tabs"]) {
                DiagnosticTab tab;
                read(t, "name", tab.name);
                read(t, "images", tab.images);
                d.tabs.push_back(std::move(tab));
            }
        if (j.contains("curves") && j["curves"].is_array())
            for (const json& c : j["curves"]) {
                DiagnosticCurve cv;
                read(c, "title", cv.title);
                cv.x = numbersFrom(c.value("x", json::array()));
                cv.y = numbersFrom(c.value("y", json::array()));
                if (c.contains("stop_x") && !c["stop_x"].is_null()) cv.stopX = numberFrom(c["stop_x"]);
                if (c.contains("labels") && c["labels"].is_array() && c["labels"].size() == 3) {
                    cv.leftLabel = c["labels"][0].get<std::string>();
                    cv.midLabel = c["labels"][1].get<std::string>();
                    cv.rightLabel = c["labels"][2].get<std::string>();
                }
                read(c, "log_y", cv.logY);
                d.curves.push_back(std::move(cv));
            }
        if (j.contains("histograms") && j["histograms"].is_array())
            for (const json& h : j["histograms"]) {
                DiagnosticHistogram hg;
                read(h, "channel", hg.channel);
                if (h.contains("color")) hg.color = colorFrom(h["color"], hg.color);
                hg.bins = numbersFrom(h.value("bins", json::array()));
                hg.binLo = numberFrom(h.value("bin_lo", json(0.0)));
                hg.binHi = numberFrom(h.value("bin_hi", json(1.0)));
                hg.lo = numberFrom(h.value("lo", json(0.0)));
                hg.hi = numberFrom(h.value("hi", json(1.0)));
                hg.gamma = numberFrom(h.value("gamma", json(1.0)));
                d.histograms.push_back(std::move(hg));
            }
        d.facts = factsFrom(j.value("facts", json::array()));
        read(j, "warnings", d.warnings);
        if (j.contains("table") && j["table"].is_object()) {
            const json& t = j["table"];
            DiagnosticTable table;
            read(t, "caption", table.caption);
            read(t, "header", table.header);
            read(t, "rows", table.rows);
            if (t.contains("accent") && t["accent"].is_array())
                for (const json& a : t["accent"])
                    if (a.is_array() && a.size() == 2) table.accentCells.emplace_back(a[0].get<int>(), a[1].get<int>());
            d.table = std::move(table);
        }
        if (j.contains("alignment") && j["alignment"].is_object()) {
            const json& a = j["alignment"];
            AlignmentInfo al;
            if (a.contains("grid") && a["grid"].is_array() && a["grid"].size() == 2) {
                al.gridRows = a["grid"][0].get<Index>();
                al.gridCols = a["grid"][1].get<Index>();
            }
            read(a, "tiles", al.tileNames);
            read(a, "highlighted", al.highlightedTile);
            al.shiftStats = factsFrom(a.value("shift_stats", json::array()));
            d.alignment = std::move(al);
        }
        return d;
    }

} // namespace sirius::app
