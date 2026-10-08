#include "core/dataset.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace sirius::app {

    std::string ChannelInfo::shortName() const {
        if (wavelengthNm > 0.0) return std::to_string(static_cast<int>(std::lround(wavelengthNm)));
        return label.empty() ? std::string("ch") : label.substr(0, 4);
    }

    std::string ChannelInfo::hexColor() const {
        char buf[8];
        auto ch = [](float v) { return static_cast<int>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
        std::snprintf(buf, sizeof buf, "#%02x%02x%02x", ch(color[0]), ch(color[1]), ch(color[2]));
        return buf;
    }

    std::array<float, 3> colorFromHex(const std::string& hex) {
        std::string h = hex;
        if (!h.empty() && h[0] == '#') h.erase(0, 1);
        if (h.size() != 6) throw std::invalid_argument("colorFromHex: expected #rrggbb, got " + hex);
        auto part = [&](std::size_t i) { return static_cast<float>(std::stoi(h.substr(i, 2), nullptr, 16)) / 255.0f; };
        return {part(0), part(2), part(4)};
    }

    std::array<float, 3> colorForWavelength(double nm) noexcept {
        // The design's channel palette; anything else is interpolated
        // between its neighbours so unusual lines still get a sensible hue.
        struct Stop {
            double nm;
            std::array<float, 3> c;
        };
        static const Stop stops[] = {
            {405.0, {0x7c / 255.f, 0x9c / 255.f, 0xff / 255.f}},
            {488.0, {0x63 / 255.f, 0xe0 / 255.f, 0x8a / 255.f}},
            {561.0, {0xe8 / 255.f, 0x71 / 255.f, 0xd9 / 255.f}},
            {640.0, {0xff / 255.f, 0x7a / 255.f, 0x5c / 255.f}},
        };
        if (nm <= 0.0) return {1.f, 1.f, 1.f};
        for (const Stop& s : stops)
            if (std::abs(s.nm - nm) < 25.0) return s.c;
        if (nm <= stops[0].nm) return stops[0].c;
        if (nm >= stops[3].nm) return stops[3].c;
        for (int i = 0; i < 3; ++i) {
            if (nm >= stops[i].nm && nm <= stops[i + 1].nm) {
                const float f = static_cast<float>((nm - stops[i].nm) / (stops[i + 1].nm - stops[i].nm));
                std::array<float, 3> c{};
                for (int k = 0; k < 3; ++k) c[static_cast<std::size_t>(k)] = stops[i].c[static_cast<std::size_t>(k)] * (1 - f) + stops[i + 1].c[static_cast<std::size_t>(k)] * f;
                return c;
            }
        }
        return {1.f, 1.f, 1.f};
    }

    void DatasetMeta::normalizeChannels() {
        const std::size_t n = static_cast<std::size_t>(std::max<Index>(dims.c, 1));
        if (rgb) {
            channels = {{"R", 0.0, {1.f, 0.f, 0.f}, {}}, {"G", 0.0, {0.f, 1.f, 0.f}, {}}, {"B", 0.0, {0.f, 0.f, 1.f}, {}}};
            return;
        }
        channels.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            ChannelInfo& ch = channels[i];
            if (ch.label.empty()) ch.label = "ch " + std::to_string(i);
            const bool defaultColor = ch.color[0] == 1.f && ch.color[1] == 1.f && ch.color[2] == 1.f;
            if (defaultColor) {
                if (ch.wavelengthNm > 0.0) ch.color = colorForWavelength(ch.wavelengthNm);
                else if (n > 1) {
                    static const double fallback[] = {488.0, 561.0, 405.0, 640.0};
                    ch.color = colorForWavelength(fallback[i % 4]);
                }
            }
        }
    }

    std::vector<std::array<double, 3>> DatasetMeta::tilePositionsPx() const {
        std::vector<std::array<double, 3>> out;
        out.reserve(tiles.size());
        // a lateral voxel size the file did not give borrows the other one
        // (pixels are square far more often than they are unknown); without
        // a z size the tiles stack in one plane
        const double vx = voxelUm[0] > 0 ? voxelUm[0] : voxelUm[1];
        const double vy = voxelUm[1] > 0 ? voxelUm[1] : voxelUm[0];
        const double vz = voxelUm[2];
        for (const TileInfo& t : tiles)
            out.push_back({vz > 0 ? t.positionUm[0] / vz : 0.0, vy > 0 ? t.positionUm[1] / vy : 0.0, vx > 0 ? t.positionUm[2] / vx : 0.0});
        return out;
    }

    std::string DatasetMeta::shapeString() const {
        if (rgb)
            return "rgb t" + std::to_string(dims.t) + " z" + std::to_string(dims.z) + " y" + std::to_string(dims.y) +
                   " x" + std::to_string(dims.x);
        return dims.toString();
    }

    std::string DatasetMeta::voxelString() const {
        char buf[96];
        std::snprintf(buf, sizeof buf, "%.3g × %.3g × %.3g µm", voxelUm[0], voxelUm[1], voxelUm[2]);
        return buf;
    }

    // --- the SIM storage layout --------------------------------------------------

    namespace {

        // The file axis a layout entry names: 0 c, 1 t, 2 z, 3 the montage (yx).
        constexpr int kMontage = 3;
        const char* const kFileAxisName[4] = {"c", "t", "z", "yx"};
        const char* const kFileAxisNoun[5] = {"channels", "time points", "sections", "rows", "columns"};
        constexpr SimAxis kOwnKind[3] = {SimAxis::C, SimAxis::T, SimAxis::Z};

        std::string lowered(std::string s) {
            for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            return s;
        }

        // An entry that says a file axis is itself: no entry at all, or its own
        // kind alone ("c=c 2"). The canonical text leaves it out -- the file's
        // own length states the extent -- and bindSimLayout puts it back so the
        // arithmetic has every axis.
        bool isIdentityEntry(const std::vector<SimFactor>& fs, int fileAxis) {
            return fs.empty() || (fs.size() == 1 && fs[0].axis == kOwnKind[fileAxis]);
        }

        std::string factorText(const SimFactor& f) {
            std::string t = simAxisName(f.axis);
            if (f.extent > 0) t += " " + std::to_string(f.extent);
            return t;
        }

        std::string factorsText(const std::vector<SimFactor>& fs) {
            std::string t;
            for (std::size_t i = 0; i < fs.size(); ++i) t += (i ? ", " : "") + factorText(fs[i]);
            return t;
        }

        // "angle 3 × phase 5 = 15": the factors with an extent and their product.
        std::string productText(const std::vector<SimFactor>& fs, Index product) {
            std::string t;
            for (const SimFactor& f : fs)
                if (f.extent > 0) t += (t.empty() ? "" : " × ") + factorText(f);
            return t + " = " + std::to_string(product);
        }

        // A tokenizer for the layout text: identifiers, integers and the
        // punctuation = ; , [ ] with whitespace between tokens ignored.
        struct Tokens {
            std::vector<std::string> items;
            std::size_t at = 0;
            explicit Tokens(const std::string& text) {
                std::size_t i = 0;
                while (i < text.size()) {
                    const unsigned char ch = static_cast<unsigned char>(text[i]);
                    if (std::isspace(ch)) {
                        ++i;
                        continue;
                    }
                    // letters and digits are separate tokens, so "3x3" and
                    // "angle3" read as "3 x 3" and "angle 3"
                    if (std::isdigit(ch)) {
                        std::size_t j = i;
                        while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
                        items.push_back(text.substr(i, j - i));
                        i = j;
                        continue;
                    }
                    if (std::isalpha(ch) || ch == '_') {
                        std::size_t j = i;
                        while (j < text.size() && (std::isalpha(static_cast<unsigned char>(text[j])) || text[j] == '_')) ++j;
                        items.push_back(lowered(text.substr(i, j - i)));
                        i = j;
                        continue;
                    }
                    if (ch == '=' || ch == ';' || ch == ',' || ch == '[' || ch == ']') {
                        items.push_back(std::string(1, text[i]));
                        ++i;
                        continue;
                    }
                    throw std::invalid_argument("SIM layout: unexpected '" + std::string(1, text[i]) + "' in \"" + text + "\"");
                }
            }
            bool done() const noexcept { return at >= items.size(); }
            const std::string& peek() const {
                static const std::string none;
                return done() ? none : items[at];
            }
            std::string next() {
                if (done()) throw std::invalid_argument("SIM layout: the text ends early");
                return items[at++];
            }
            void expect(const char* tok) {
                const std::string got = done() ? "the end" : "'" + items[at] + "'";
                if (done() || items[at] != tok) throw std::invalid_argument(std::string("SIM layout: expected '") + tok + "', got " + got);
                ++at;
            }
            static bool isInteger(const std::string& s) {
                return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
            }
        };

        std::optional<SimAxis> kindNamed(const std::string& s) {
            if (s == "angle" || s == "angles" || s == "dir" || s == "dirs" || s == "direction" || s == "directions") return SimAxis::Angle;
            if (s == "phase" || s == "phases") return SimAxis::Phase;
            if (s == "z") return SimAxis::Z;
            if (s == "c") return SimAxis::C;
            if (s == "t") return SimAxis::T;
            return std::nullopt;
        }

        // kind [extent]
        SimFactor parseFactor(Tokens& tk) {
            const std::string name = tk.next();
            const std::optional<SimAxis> kind = kindNamed(name);
            if (!kind) throw std::invalid_argument("SIM layout: '" + name + "' is not an axis (angle, phase, z, c or t)");
            SimFactor f;
            f.axis = *kind;
            if (Tokens::isInteger(tk.peek())) {
                f.extent = static_cast<Index>(std::stoll(tk.next()));
                if (f.extent < 1) throw std::invalid_argument("SIM layout: " + name + " needs an extent of at least 1");
            }
            return f;
        }

        // factor | [factor, factor, ...]
        std::vector<SimFactor> parseFactors(Tokens& tk) {
            std::vector<SimFactor> fs;
            if (tk.peek() == "[") {
                tk.next();
                for (;;) {
                    fs.push_back(parseFactor(tk));
                    if (tk.peek() == ",") {
                        tk.next();
                        continue;
                    }
                    tk.expect("]");
                    break;
                }
            } else {
                fs.push_back(parseFactor(tk));
            }
            return fs;
        }

        // The checks shared by the parser and the shorthand: one assignment
        // per logical axis, the kinds on the axes that can hold them, the
        // extents that have to be written down, angle or phase named.
        void checkStorage(const SimStorage& st) {
            int seen[5] = {0, 0, 0, 0, 0};
            auto note = [&](const SimFactor& f, int fileAxis) {
                const int k = static_cast<int>(f.axis);
                if (seen[k]++) throw std::invalid_argument(std::string("SIM layout: ") + simAxisName(f.axis) + " is assigned twice");
                const bool own = fileAxis < kMontage && f.axis == kOwnKind[fileAxis];
                if ((f.axis == SimAxis::C || f.axis == SimAxis::T) && !own)
                    throw std::invalid_argument(std::string("SIM layout: ") + simAxisName(f.axis) + " can only stand on the " + simAxisName(f.axis) + " axis");
                if (f.extent == 0 && !own)
                    throw std::invalid_argument(std::string("SIM layout: ") + simAxisName(f.axis) + " on " + kFileAxisName[fileAxis] +
                                                " needs its extent (how many " + simAxisName(f.axis) + "s), e.g. " + simAxisName(f.axis) + " 3");
            };
            for (int a = 0; a < 3; ++a) {
                if (st.axes[static_cast<std::size_t>(a)].empty()) continue;
                for (const SimFactor& f : st.axes[static_cast<std::size_t>(a)]) note(f, a);
            }
            for (const SimFactor& f : st.tiles) note(f, kMontage);
            if (st.rows < 1 || st.cols < 1) throw std::invalid_argument("SIM layout: the montage grid needs at least 1 x 1 tiles");
            if (!seen[static_cast<int>(SimAxis::Angle)] && !seen[static_cast<int>(SimAxis::Phase)])
                throw std::invalid_argument("SIM layout: a layout names angle or phase at least once");
        }

        Index productOfKnown(const std::vector<SimFactor>& fs) {
            Index p = 1;
            for (const SimFactor& f : fs)
                if (f.extent > 0) p *= f.extent;
            return p;
        }

        // The coordinate of a logical frame on one kind of axis.
        Index coordinate(const SimFrames::Logical& l, SimAxis a) noexcept {
            switch (a) {
                case SimAxis::Angle: return l.angle;
                case SimAxis::Phase: return l.phase;
                case SimAxis::Z: return l.z;
                case SimAxis::C: return l.c;
                case SimAxis::T: return l.t;
            }
            return 0;
        }
        Index& coordinate(SimFrames::Logical& l, SimAxis a) noexcept {
            switch (a) {
                case SimAxis::Angle: return l.angle;
                case SimAxis::Phase: return l.phase;
                case SimAxis::Z: return l.z;
                case SimAxis::C: return l.c;
                case SimAxis::T: return l.t;
            }
            return l.z;
        }

        // index = sum of coordinate * stride, the first factor outermost.
        Index composeIndex(const std::vector<SimFactor>& fs, const SimFrames::Logical& l, const char* where) {
            Index index = 0;
            for (const SimFactor& f : fs) {
                const Index k = coordinate(l, f.axis);
                if (k < 0 || k >= f.extent)
                    throw std::out_of_range(std::string("SIM layout: ") + simAxisName(f.axis) + " " + std::to_string(k) + " is outside " +
                                            factorText(f) + " on " + where);
                index = index * f.extent + k;
            }
            return index;
        }

        void decomposeIndex(const std::vector<SimFactor>& fs, Index index, SimFrames::Logical& l, const char* where) {
            Index total = 1;
            for (const SimFactor& f : fs) total *= f.extent;
            if (index < 0 || index >= total)
                throw std::out_of_range(std::string("SIM layout: index ") + std::to_string(index) + " is outside the " + std::to_string(total) + " on " + where);
            for (auto it = fs.rbegin(); it != fs.rend(); ++it) {
                coordinate(l, it->axis) = index % it->extent;
                index /= it->extent;
            }
        }

    } // namespace

    const char* simAxisName(SimAxis a) noexcept {
        switch (a) {
            case SimAxis::Angle: return "angle";
            case SimAxis::Phase: return "phase";
            case SimAxis::Z: return "z";
            case SimAxis::C: return "c";
            case SimAxis::T: return "t";
        }
        return "?";
    }

    std::string SimStorage::text() const {
        std::string t;
        auto entry = [&](const char* axis, const std::string& body) {
            if (!t.empty()) t += "; ";
            t += std::string(axis) + "=" + body;
        };
        for (int a = 0; a < 3; ++a) {
            const std::vector<SimFactor>& fs = axes[static_cast<std::size_t>(a)];
            // the identity is what leaving the entry out says, so a bound
            // layout reads back as the layout that was written down
            if (isIdentityEntry(fs, a)) continue;
            entry(kFileAxisName[a], fs.size() == 1 ? factorText(fs[0]) : "[" + factorsText(fs) + "]");
        }
        if (montage()) entry("yx", std::to_string(rows) + "x" + std::to_string(cols) + "[" + factorsText(tiles) + "]");
        return t;
    }

    Index SimStorage::angles() const noexcept {
        for (const auto& fs : axes)
            for (const SimFactor& f : fs)
                if (f.axis == SimAxis::Angle) return f.extent;
        for (const SimFactor& f : tiles)
            if (f.axis == SimAxis::Angle) return f.extent;
        return 1;
    }

    Index SimStorage::phases() const noexcept {
        for (const auto& fs : axes)
            for (const SimFactor& f : fs)
                if (f.axis == SimAxis::Phase) return f.extent;
        for (const SimFactor& f : tiles)
            if (f.axis == SimAxis::Phase) return f.extent;
        return 1;
    }

    SimStorage parseSimStorage(const std::string& text) {
        Tokens tk(text);
        if (tk.done()) throw std::invalid_argument("SIM layout: empty");
        SimStorage st;
        bool given[4] = {false, false, false, false};
        for (;;) {
            const std::string axis = tk.next();
            int which = -1;
            for (int a = 0; a < 4; ++a)
                if (axis == kFileAxisName[a]) which = a;
            if (which < 0) throw std::invalid_argument("SIM layout: '" + axis + "' is not a file axis (c, t, z or yx)");
            if (given[which]) throw std::invalid_argument("SIM layout: " + axis + " is given twice");
            given[which] = true;
            tk.expect("=");
            if (which == kMontage) {
                // RxC before the bracket, or the first factor's extent by the rest
                std::optional<Index> rows, cols;
                if (Tokens::isInteger(tk.peek())) {
                    rows = static_cast<Index>(std::stoll(tk.next()));
                    const std::string x = tk.next();
                    if (x != "x" || !Tokens::isInteger(tk.peek()))
                        throw std::invalid_argument("SIM layout: the montage grid is written rows x cols, e.g. yx=3x3[angle 3, phase 3]");
                    cols = static_cast<Index>(std::stoll(tk.next()));
                }
                if (tk.peek() != "[") throw std::invalid_argument("SIM layout: a montage lists its factors in brackets, e.g. yx=3x3[angle 3, phase 3]");
                st.tiles = parseFactors(tk);
                for (const SimFactor& f : st.tiles)
                    if (f.extent == 0)
                        throw std::invalid_argument(std::string("SIM layout: ") + simAxisName(f.axis) + " in the montage needs its extent");
                if (rows) {
                    st.rows = *rows;
                    st.cols = *cols;
                } else {
                    st.rows = st.tiles.front().extent;
                    st.cols = 1;
                    for (std::size_t i = 1; i < st.tiles.size(); ++i) st.cols *= st.tiles[i].extent;
                }
                const Index n = productOfKnown(st.tiles);
                if (st.rows * st.cols != n)
                    throw std::invalid_argument("SIM layout: the montage " + std::to_string(st.rows) + "x" + std::to_string(st.cols) + " has " +
                                                std::to_string(st.rows * st.cols) + " tiles, but " + productText(st.tiles, n) + ".");
            } else {
                st.axes[static_cast<std::size_t>(which)] = parseFactors(tk);
            }
            if (tk.done()) break;
            tk.expect(";");
            if (tk.done()) break;   // a trailing ';'
        }
        checkStorage(st);
        return st;
    }

    SimStorage simStorageOf(const SimLayout& layout) {
        if (!layout.storage.empty()) return parseSimStorage(layout.storage);
        SimStorage st;
        const SimFactor angle{SimAxis::Angle, std::max(layout.ndirs, 1)};
        const SimFactor z{SimAxis::Z, 0};
        const SimFactor phase{SimAxis::Phase, std::max(layout.nphases, 1)};
        st.axes[2] = layout.fastSi ? std::vector<SimFactor>{z, angle, phase} : std::vector<SimFactor>{angle, z, phase};
        return st;
    }

    std::string SimLayout::text() const { return storage.empty() ? simStorageOf(*this).text() : storage; }

    SimLayout SimLayout::fromText(const std::string& text) {
        const SimStorage st = parseSimStorage(text);
        SimLayout l;
        l.present = true;
        l.storage = st.text();
        l.ndirs = static_cast<int>(st.angles());
        l.nphases = static_cast<int>(st.phases());
        // the fast-SI flag mirrors a z-packed layout in that order, so a reader
        // of the shorthand fields sees the same stack the storage describes
        const std::vector<SimFactor>& z = st.axes[2];
        l.fastSi = !st.montage() && isIdentityEntry(st.axes[0], 0) && isIdentityEntry(st.axes[1], 1) && z.size() == 3 &&
                   z[0].axis == SimAxis::Z && z[1].axis == SimAxis::Angle && z[2].axis == SimAxis::Phase;
        return l;
    }

    SimLayout SimLayout::shorthand(int ndirs, int nphases, bool fastSi) {
        SimLayout l;
        l.present = true;
        l.ndirs = ndirs;
        l.nphases = nphases;
        l.fastSi = fastSi;
        return l;
    }

    SimFrames bindSimLayout(const SimLayout& layout, const Dims5& dims) {
        SimFrames f;
        f.dims = dims;
        f.storage = simStorageOf(layout);
        SimStorage& st = f.storage;
        for (int a = 0; a < 3; ++a) {
            std::vector<SimFactor>& fs = st.axes[static_cast<std::size_t>(a)];
            if (fs.empty()) fs = {SimFactor{kOwnKind[a], 0}};   // the axis is itself
            const Index length = dims[kAxes[static_cast<std::size_t>(a)]];
            const Index known = productOfKnown(fs);
            auto remainder = std::find_if(fs.begin(), fs.end(), [](const SimFactor& x) { return x.extent == 0; });
            const std::string holds = std::string(kFileAxisName[a]) + " holds " + std::to_string(length) + " " + kFileAxisNoun[a];
            if (remainder != fs.end()) {
                if (known <= 0 || length % known != 0)
                    throw std::invalid_argument(holds + ", not a multiple of " + productText(fs, known) + ".");
                remainder->extent = length / known;
            } else if (known != length) {
                throw std::invalid_argument(holds + ", but the layout needs " + productText(fs, known) + ".");
            }
        }
        if (st.montage()) {
            if (dims.y % st.rows != 0)
                throw std::invalid_argument("y holds " + std::to_string(dims.y) + " rows, not a multiple of " + std::to_string(st.rows) + " tile rows.");
            if (dims.x % st.cols != 0)
                throw std::invalid_argument("x holds " + std::to_string(dims.x) + " columns, not a multiple of " + std::to_string(st.cols) + " tile columns.");
            f.rows = st.rows;
            f.cols = st.cols;
        }
        f.tileY = dims.y / f.rows;
        f.tileX = dims.x / f.cols;
        auto extentOf = [&](SimAxis kind, Index fallback) {
            for (const auto& fs : st.axes)
                for (const SimFactor& x : fs)
                    if (x.axis == kind) return x.extent;
            for (const SimFactor& x : st.tiles)
                if (x.axis == kind) return x.extent;
            return fallback;
        };
        f.angles = extentOf(SimAxis::Angle, 1);
        f.phases = extentOf(SimAxis::Phase, 1);
        f.nz = extentOf(SimAxis::Z, 1);
        f.channels = extentOf(SimAxis::C, 1);
        f.times = extentOf(SimAxis::T, 1);
        return f;
    }

    std::string simLayoutProblem(const SimLayout& layout, const Dims5& dims) {
        try {
            bindSimLayout(layout, dims);
            return {};
        } catch (const std::exception& e) {
            return e.what();
        }
    }

    SimFrames::Frame SimFrames::frameOf(const Logical& l) const {
        Frame fr;
        fr.c = composeIndex(storage.axes[0], l, "c");
        fr.t = composeIndex(storage.axes[1], l, "t");
        fr.z = composeIndex(storage.axes[2], l, "z");
        if (storage.montage()) {
            const Index tile = composeIndex(storage.tiles, l, "yx");
            fr.row = tile / cols;
            fr.col = tile % cols;
        }
        return fr;
    }

    SimFrames::Logical SimFrames::logicalOf(const Frame& fr) const {
        Logical l;
        decomposeIndex(storage.axes[0], fr.c, l, "c");
        decomposeIndex(storage.axes[1], fr.t, l, "t");
        decomposeIndex(storage.axes[2], fr.z, l, "z");
        if (storage.montage()) {
            if (fr.row < 0 || fr.row >= rows || fr.col < 0 || fr.col >= cols)
                throw std::out_of_range("SIM layout: tile (" + std::to_string(fr.row) + ", " + std::to_string(fr.col) + ") is outside the " +
                                        std::to_string(rows) + "x" + std::to_string(cols) + " montage");
            decomposeIndex(storage.tiles, fr.row * cols + fr.col, l, "yx");
        }
        return l;
    }

    std::optional<bool> SimFrames::libraryOrder() const {
        if (storage.montage() || channels != dims.c || times != dims.t || sections() != dims.z) return std::nullopt;
        // on c and t only their own kind; every other factor on z
        for (int a = 0; a < 2; ++a)
            for (const SimFactor& x : storage.axes[static_cast<std::size_t>(a)])
                if (x.axis != kOwnKind[a]) return std::nullopt;
        // the two orders the reconstructor reads, checked frame by frame
        bool slow = true, fast = true;
        for (Index d = 0; d < angles; ++d)
            for (Index z = 0; z < nz; ++z)
                for (Index p = 0; p < phases; ++p) {
                    const Index s = sectionIndex(d, p, z);
                    slow = slow && s == (d * nz + z) * phases + p;
                    fast = fast && s == (z * angles + d) * phases + p;
                }
        if (slow) return false;
        if (fast) return true;
        return std::nullopt;
    }

} // namespace sirius::app
