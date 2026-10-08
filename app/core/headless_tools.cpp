#include "core/headless.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "core/help_pages.hpp"

// The pure halves of the headless tools: DatasetInfo, the open options, the
// help pages, the operation descriptions and the diagnostics as JSON. The
// tool table itself, which needs the workbench, is in headless.cpp.

namespace sirius::app {

    using json = nlohmann::json;

    namespace {
        namespace fs = std::filesystem;

        [[noreturn]] void invalid(const std::string& message, const std::string& hint = {}) {
            throw ToolFailure("invalid_argument", message, hint);
        }

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        // The path as the reply reports it: absolute, forward slashes. A path
        // that does not convert (bytes that are not UTF-8) is left as given.
        std::string reportedPath(const std::string& path) {
            if (path.empty()) return path;
            try {
                std::error_code ec;
                fs::path p = fs::absolute(fs::u8path(path), ec);
                if (ec) p = fs::u8path(path);
                return p.lexically_normal().generic_u8string();
            } catch (const std::exception&) {
                return path;
            }
        }

        // An integer argument: an integer, or a number with nothing after the
        // point (a model writes 2 as 2.0 now and then).
        std::int64_t integerArg(const json& a, const char* key, std::int64_t def, std::int64_t min) {
            const auto it = a.find(key);
            if (it == a.end() || it->is_null()) return def;
            if (!it->is_number() || (!it->is_number_integer() && it->get<double>() != std::floor(it->get<double>())))
                invalid(std::string("'") + key + "' must be an integer");
            const std::int64_t v = it->is_number_integer() ? it->get<std::int64_t>() : static_cast<std::int64_t>(it->get<double>());
            if (v < min) invalid(std::string("'") + key + "' must be at least " + std::to_string(min));
            return v;
        }

        bool boolArg(const json& a, const char* key, bool def) {
            const auto it = a.find(key);
            if (it == a.end() || it->is_null()) return def;
            if (!it->is_boolean()) invalid(std::string("'") + key + "' must be true or false");
            return it->get<bool>();
        }

        const char* typeName(ParamType t) noexcept {
            switch (t) {
                case ParamType::Bool: return "bool";
                case ParamType::Int: return "int";
                case ParamType::Double: return "double";
                case ParamType::String: return "string";
                case ParamType::Path: return "path";
                case ParamType::Choice: return "choice";
                case ParamType::Channel: return "channel";
                case ParamType::Axes: return "axes";
                case ParamType::DoubleList: return "double_list";
                case ParamType::StringList: return "string_list";
                case ParamType::Prompts: return "prompts";
            }
            return "string";
        }

        // Infinite bounds are "none", which JSON spells null.
        json bound(double v) { return std::isfinite(v) ? json(v) : json(nullptr); }

        // What loadHelpPage returns for a kind that has no page (help_pages.cpp):
        // a placeholder that asks the user to write one.
        bool helpPageExists(const HelpPage& page) {
            std::error_code ec;
            if (!page.path.empty() && fs::is_regular_file(fs::u8path(page.path), ec)) return true;
            // a page registered in memory (a plugin's docstring) has no file
            return page.intro != "No help page yet \xE2\x80\x94 click Edit page to write one.";
        }

        // A plugin's kind may have a '.' in it ("user.denoise"), which names
        // a file in the help directory as safely: a registered kind is
        // allowed as long as it cannot leave that directory (ToolApi's rule).
        bool pageNameAllowed(const std::string& name) {
            if (isHelpPageName(name)) return true;
            if (name.empty() || name.front() == '.' || name.size() > 128) return false;
            for (const char c : name)
                if (c == '/' || c == '\\' || c == ':' || c == '\0') return false;
            return findOperation(name) != nullptr;
        }

        // At most `n` of the values, evenly spaced, the last one kept.
        std::vector<double> decimated(const std::vector<double>& v, std::size_t n) {
            if (v.size() <= n || n < 2) return v;
            std::vector<double> out;
            out.reserve(n);
            for (std::size_t i = 0; i < n; ++i) out.push_back(v[i * (v.size() - 1) / (n - 1)]);
            return out;
        }
    } // namespace

    // --- datasets --------------------------------------------------------------------

    // The SIM layout of a DatasetInfo: {present: false} for a stack that is
    // not SIM -- the angle and phase counts beside it were the struct's
    // defaults and read as a detection -- else the counts, the order and,
    // for the general form, its layout text.
    json simInfo(const SimLayout& sim) {
        json j = {{"present", sim.present}};
        if (!sim.present) return j;
        j["ndirs"] = sim.ndirs;
        j["nphases"] = sim.nphases;
        j["fast"] = sim.fastSi;
        if (!sim.isShorthand()) j["layout"] = sim.storage;
        return j;
    }

    json datasetInfo(const DatasetMeta& meta, const OpenResult* opened) {
        json dims = {{"c", meta.dims.c}, {"t", meta.dims.t}, {"z", meta.dims.z}, {"y", meta.dims.y}, {"x", meta.dims.x}};
        json float32Bytes = nullptr;
        try {
            float32Bytes = static_cast<std::uint64_t>(meta.dims.numel()) * 4u;
        } catch (const std::exception&) {
            // extents whose product overflows: the size is not a number
        }
        json channels = json::array();
        for (std::size_t c = 0; c < meta.channels.size(); ++c) {
            const ChannelInfo& ch = meta.channels[c];
            channels.push_back({{"index", c}, {"label", ch.label}, {"wavelength_nm", ch.wavelengthNm}, {"color", ch.hexColor()}});
        }
        json tiles = json::array();
        for (std::size_t i = 0; i < meta.tiles.size(); ++i) {
            const TileInfo& t = meta.tiles[i];
            tiles.push_back({{"index", i}, {"name", t.name}, {"position_um", {t.positionUm[0], t.positionUm[1], t.positionUm[2]}}});
        }
        return {{"path", reportedPath(meta.sourcePath)},
                {"name", meta.name},
                {"format", meta.format},
                {"shape", meta.shapeString()},
                {"dims", dims},
                {"dtype", toString(meta.sourceType)},
                {"bytes_on_disk", meta.bytesOnDisk},
                {"float32_bytes", float32Bytes},
                {"voxel_um", {meta.voxelUm[0], meta.voxelUm[1], meta.voxelUm[2]}},
                {"frame_interval_s", meta.frameIntervalS},
                {"channels", channels},
                {"acquisition", meta.acquisition},
                {"sim", simInfo(meta.sim)},
                {"rgb", meta.rgb},
                {"light_sheet", meta.lightSheet},
                {"sheet_angle_deg", meta.sheetAngleDeg},
                {"tiles", tiles},
                {"tile", meta.tileIndex},
                {"metadata_summary", opened ? opened->metadataSummary : std::string()},
                // Only opening the file finds that out: without it the answer is
                // unknown (null), not "no".
                {"dims_from_metadata", opened ? json(opened->dimsFromMetadata) : json(nullptr)},
                {"full_load_skipped", opened ? opened->fullLoadSkipped : std::string()}};
    }

    OpenOptions openOptionsFromJson(const json& a) {
        OpenOptions o;
        if (!a.is_object()) return o;
        if (a.contains("page_order") || a.contains("c") || a.contains("t") || a.contains("z")) {
            PageOrder po;
            if (const auto it = a.find("page_order"); it != a.end() && !it->is_null()) {
                if (!it->is_string()) invalid("'page_order' must be a string such as \"czt\"");
                po.order = lower(it->get<std::string>());
                std::string sorted = po.order;
                std::sort(sorted.begin(), sorted.end());
                if (sorted != "ctz")
                    invalid("'page_order' must name c, z and t once each, fastest first (\"czt\", \"zct\", ...)");
            }
            po.c = static_cast<Index>(integerArg(a, "c", 1, 1));
            po.t = static_cast<Index>(integerArg(a, "t", 1, 1));
            po.z = static_cast<Index>(integerArg(a, "z", 0, 0));   // 0: from the page count
            o.pageOrder = po;
        }
        if (const auto it = a.find("voxel_um"); it != a.end() && !it->is_null()) {
            if (!it->is_array() || it->size() != 3) invalid("'voxel_um' must be [x, y, z] in micrometres");
            std::array<double, 3> v{};
            for (std::size_t i = 0; i < 3; ++i) {
                if (!(*it)[i].is_number() || !((*it)[i].get<double>() > 0.0)) invalid("'voxel_um' entries must be positive numbers");
                v[i] = (*it)[i].get<double>();
            }
            o.voxelUm = v;
        }
        if (const auto it = a.find("sim"); it != a.end() && !it->is_null()) {
            SimLayout sim;
            // the general form: a layout text, alone or as {"layout": ...}
            auto fromLayoutText = [&](const std::string& text) {
                try {
                    sim = SimLayout::fromText(text);
                } catch (const std::exception& e) {
                    invalid(std::string("'sim' layout: ") + e.what(), "a layout such as \"z=[angle 3, z, phase 5]\", \"c=angle 3; z=phase 3\" or \"yx=3x3[angle 3, phase 3]\"");
                }
            };
            if (it->is_boolean()) {
                sim.present = it->get<bool>();   // true: 3 directions x 5 phases
            } else if (it->is_string()) {
                fromLayoutText(it->get<std::string>());
            } else if (it->is_object() && it->contains("layout") && !(*it)["layout"].is_null()) {
                if (!(*it)["layout"].is_string()) invalid("'sim.layout' must be a string");
                fromLayoutText((*it)["layout"].get<std::string>());
            } else if (it->is_object()) {
                sim.present = true;
                sim.ndirs = static_cast<int>(integerArg(*it, "ndirs", 3, 1));
                sim.nphases = static_cast<int>(integerArg(*it, "nphases", 5, 1));
                sim.fastSi = boolArg(*it, "fast", false);
            } else if (it->is_array() && (it->size() == 2 || it->size() == 3)) {
                // [ndirs, nphases(, fast)], the CLI's --sim d,p[,fast] as JSON
                const json named = {{"ndirs", (*it)[0]}, {"nphases", (*it)[1]}, {"fast", it->size() == 3 ? (*it)[2] : json(false)}};
                sim.present = true;
                sim.ndirs = static_cast<int>(integerArg(named, "ndirs", 3, 1));
                sim.nphases = static_cast<int>(integerArg(named, "nphases", 5, 1));
                sim.fastSi = named["fast"].is_boolean() ? named["fast"].get<bool>() : named["fast"].is_number() && named["fast"].get<double>() != 0.0;
            } else {
                invalid("'sim' must be {\"ndirs\", \"nphases\", \"fast\"}, {\"layout\": \"...\"}, a layout text or false");
            }
            o.sim = sim;
        }
        o.tile = static_cast<Index>(integerArg(a, "tile", 0, 0));
        o.readAll = boolArg(a, "full_load", false);
        return o;
    }

    // --- help pages --------------------------------------------------------------------

    bool isHelpPageName(const std::string& name) noexcept {
        if (name.empty() || name.size() > 128) return false;
        for (const char c : name)
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
        return true;
    }

    json helpPageJson(const std::string& page, std::size_t maxChars) {
        if (!pageNameAllowed(page))
            invalid("'" + page + "' is not a help page name (letters, digits, '_' and '-' only)",
                    "get_help without arguments lists the pages");
        const HelpPage p = loadHelpPage(page);
        if (!helpPageExists(p))
            throw ToolFailure("not_found", "there is no help page '" + page + "'", "get_help without arguments lists the pages");
        std::string markdown = p.markdown;
        bool truncated = false;
        if (markdown.size() > maxChars) {
            std::size_t cut = maxChars;
            while (cut > 0 && (static_cast<unsigned char>(markdown[cut]) & 0xC0) == 0x80) --cut;
            markdown.resize(cut);
            truncated = true;
        }
        std::error_code ec;
        const bool onDisk = !p.path.empty() && fs::is_regular_file(fs::u8path(p.path), ec);
        return {{"page", page},
                {"title", p.title},
                {"path", onDisk ? reportedPath(p.path) : std::string()},
                {"exists", true},
                {"markdown", markdown},
                {"truncated", truncated}};
    }

    json helpPageList() {
        std::vector<std::pair<std::string, std::string>> pages;
        std::error_code ec;
        const fs::path dir = fs::u8path(helpDirectory());
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code fec;
            if (!it->is_regular_file(fec) || it->path().extension() != ".md") continue;
            const std::string stem = it->path().stem().u8string();
            if (!isHelpPageName(stem)) continue;
            pages.emplace_back(stem, loadHelpPage(stem).title);
        }
        std::sort(pages.begin(), pages.end());
        json list = json::array();
        for (const auto& [page, title] : pages) list.push_back({{"page", page}, {"title", title}});
        return {{"pages", list}};
    }

    // --- operations ----------------------------------------------------------------------

    json operationJson(const Operation& op, bool detail) {
        const OpInfo& info = op.info();
        // A list either way, of the keys or of the parameters themselves, so
        // that the field has one shape for a reader.
        json params = json::array();
        for (const ParamSpec& s : info.params)
            if (detail) params.push_back({{"key", s.key}, {"label", s.label}, {"default", toJson(s.defaultValue)}, {"schema", schemaOf(s)}});
            else params.push_back(s.key);
        json presets = json::array();
        for (const ParamPreset& p : info.presets) presets.push_back(p.name);
        bool needsWorker = info.remoteCapable;
        try {
            needsWorker = op.needsWorker(op.defaults());
        } catch (const std::exception&) {
            // an operation that cannot say without its input: what it could need
        }
        return {{"kind", info.kind},
                {"name", info.name},
                {"group", info.group},
                {"params", params},
                {"presets", presets},
                {"produces_labels", info.producesLabels},
                {"needs_labels", info.needsLabels},
                {"needs_worker", needsWorker},
                {"plugin", info.plugin},
                {"gpu", info.hasGpuPath}};
    }

    json describeOperation(const Operation& op) {
        const OpInfo& info = op.info();
        json params = json::array();
        for (const ParamSpec& s : info.params) {
            json p = {{"key", s.key},
                      {"label", s.label},
                      {"type", typeName(s.type)},
                      {"default", toJson(s.defaultValue)},
                      {"choices", s.choices},
                      {"min", bound(s.min)},
                      {"max", bound(s.max)},
                      {"unit", s.unit},
                      {"advanced", s.advanced},
                      {"schema", schemaOf(s)}};
            if (!s.help.empty()) p["help"] = s.help;
            params.push_back(std::move(p));
        }
        json presets = json::array();
        for (const ParamPreset& preset : info.presets) {
            json values = json::object();
            for (const auto& [key, value] : preset.values) values[key] = toJson(value);
            presets.push_back({{"name", preset.name}, {"summary", preset.summary}, {"values", values}});
        }
        json help = nullptr;
        try {
            const HelpPage page = loadHelpPage(info.helpPage.empty() ? info.kind : info.helpPage);
            if (helpPageExists(page)) help = {{"title", page.title}, {"intro", page.intro}};
        } catch (const std::exception&) {
            // no page is no help, not a failure to describe the operation
        }
        json out = operationJson(op, false);
        out["params"] = params;
        out["presets"] = presets;
        out["help"] = help;
        return out;
    }

    // --- diagnostics ---------------------------------------------------------------------

    json diagnosticsJson(const Diagnostics& d, int stepIndex, bool detail) {
        json j = {{"step", stepIndex + 1}, {"summary", d.summary}, {"footer", d.footer}, {"warnings", d.warnings}};
        json facts = json::object();
        for (const DiagnosticFact& f : d.facts) facts[f.key] = f.value;
        j["facts"] = facts;
        if (d.table) j["table"] = {{"caption", d.table->caption}, {"header", d.table->header}, {"rows", d.table->rows}};
        json curves = json::array();
        for (const DiagnosticCurve& c : d.curves) {
            json cj = {{"title", c.title}, {"points", c.y.size()}};
            if (!c.y.empty()) {
                cj["first"] = c.y.front();
                cj["last"] = c.y.back();
            }
            if (detail) {
                cj["x"] = decimated(c.x, 200);
                cj["y"] = decimated(c.y, 200);
                cj["log_y"] = c.logY;
                if (c.stopX) cj["stop_x"] = *c.stopX;
            }
            curves.push_back(std::move(cj));
        }
        j["curves"] = curves;
        json hists = json::array();
        for (const DiagnosticHistogram& h : d.histograms) {
            json hj = {{"channel", h.channel}, {"lo", h.lo}, {"hi", h.hi}, {"gamma", h.gamma}};
            if (detail) {
                hj["bins"] = h.bins;
                hj["bin_lo"] = h.binLo;
                hj["bin_hi"] = h.binHi;
            }
            hists.push_back(std::move(hj));
        }
        j["histograms"] = hists;
        json tabs = json::array();
        for (const DiagnosticTab& t : d.tabs) tabs.push_back(t.name);
        j["tabs"] = tabs;
        // What render_diagnostics can draw, and under which tab each image is
        // shown (no tabs: a single "Preview" tab of every image).
        json images = json::array();
        for (std::size_t i = 0; i < d.images.size(); ++i) {
            const DiagnosticImage& img = d.images[i];
            std::string tab = d.tabs.empty() ? "Preview" : std::string();
            for (const DiagnosticTab& t : d.tabs)
                if (tab.empty() && std::find(t.images.begin(), t.images.end(), static_cast<int>(i)) != t.images.end()) tab = t.name;
            images.push_back({{"index", i},
                              {"title", img.title},
                              {"meta", img.meta},
                              {"rows", img.rows},
                              {"cols", img.cols},
                              {"log_scale", img.logScale},
                              {"marks", img.marks.size()},
                              {"tab", tab}});
        }
        j["images"] = images;
        return j;
    }

} // namespace sirius::app
