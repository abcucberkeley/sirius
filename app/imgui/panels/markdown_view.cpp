#include "imgui/panels/markdown_view.hpp"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <utility>

#include "core/help_pages.hpp"
#include "imgui/platform.hpp"
#include "imgui/strings.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui::markdown {

    namespace fs = std::filesystem;
    using theme::px;
    using theme::Weight;

    namespace {

        // Slant of the italics, as a fraction of the height. Archivo is
        // bundled upright only, so an italic is the upright face sheared
        // when it is drawn -- about the pivot below, so that a slanted letter
        // stays centred on the room measured for the upright one.
        constexpr float kSlant = 0.18f;
        constexpr float kSlantPivot = 0.3f;     // above the baseline, in font sizes
        // Sub- and superscripts: smaller, on a shifted baseline.
        constexpr float kScriptScale = 0.72f;
        constexpr float kSupRise = 0.38f;       // of the size of the text they are attached to
        constexpr float kSubDrop = 0.20f;
        // Height of the fraction rule above the baseline (the "math axis").
        constexpr float kAxis = 0.27f;

        float displaySize(float designPx) {
            const ImGuiStyle& st = ImGui::GetStyle();
            return designPx * st.FontScaleMain * st.FontScaleDpi;
        }

        // --- styles ---------------------------------------------------------------

        // What the parsers carry around: the style asked for plus what the
        // markup added so far.
        struct State {
            TextStyle base;
            float scale = 1.0f;          // scripts
            float rise = 0.0f;           // baseline shift upwards, display pixels
            bool overline = false;
            bool underline = false;
            bool code = false;           // inline code: surface fill behind it
            bool keepSpaces = false;     // preformatted: every space counts
            std::string link;
        };

        // A style resolved to a font of this display.
        struct Run {
            ImFont* font = nullptr;
            float size = 0.0f;           // display pixels
            float ascent = 0.0f;         // of the font at that size
            ImU32 color = theme::kText;
            ImU32 background = theme::kTransparent;
            bool italic = false;
            bool overline = false;
            bool underline = false;
            float rise = 0.0f;
            std::string link;
        };

        ImFont* faceOf(const TextStyle& s) {
            ImFont* f = nullptr;
            switch (s.face) {
                case Face::Mono: f = theme::mono(); break;
                case Face::Caption: f = theme::captionFont(); break;
                case Face::Body: f = theme::font(s.weight); break;
            }
            return f ? f : ImGui::GetFont();
        }

        Run resolve(const State& st) {
            Run r;
            r.font = faceOf(st.base);
            r.size = std::max(1.0f, displaySize(st.base.px * st.scale));
            ImFontBaked* baked = r.font->GetFontBaked(r.size);
            r.ascent = baked ? baked->Ascent * (baked->Size > 0.0f ? r.size / baked->Size : 1.0f) : r.size * 0.8f;
            r.color = st.base.color;
            r.background = st.code ? theme::kSurface : theme::kTransparent;
            r.italic = st.base.italic;
            r.overline = st.overline;
            r.underline = st.underline;
            r.rise = st.rise;
            r.link = st.link;
            return r;
        }

        float measure(const Run& r, const std::string& text) {
            if (text.empty()) return 0.0f;
            return r.font->CalcTextSizeA(r.size, FLT_MAX, 0.0f, text.c_str(), text.c_str() + text.size()).x;
        }

        // --- boxes ----------------------------------------------------------------

        struct Box {
            float w = 0.0f;
            float asc = 0.0f;      // above the baseline
            float desc = 0.0f;     // below it
            Box() = default;
            Box(const Box&) = delete;
            Box& operator=(const Box&) = delete;
            virtual ~Box() = default;
            virtual void emit(Layout& out, float x, float baseline) const = 0;
        };
        using BoxPtr = std::unique_ptr<Box>;

        bool accentOf(char32_t c, Layout::Accent& kind) {
            switch (c) {
                case 0x0303: kind = Layout::Accent::Tilde; return true;
                case 0x0302: kind = Layout::Accent::Hat; return true;
                case 0x0304:
                case 0x0305: kind = Layout::Accent::Bar; return true;
                case 0x20D7: kind = Layout::Accent::Vec; return true;
                case 0x0307: kind = Layout::Accent::Dot; return true;
                case 0x0308: kind = Layout::Accent::DDot; return true;
                default: return false;
            }
        }

        struct TextBox final : Box {
            std::string text;
            Run run;
            bool accented = false;
            Layout::Accent accent = Layout::Accent::Tilde;
            float glyphTop = 0.0f;     // of the accented glyph, below the top of the line box

            TextBox(std::string t, const Run& r) : text(std::move(t)), run(r) {
                w = measure(run, text);
                asc = run.ascent + run.rise;
                desc = (run.size - run.ascent) - run.rise;
            }

            // A mark drawn over the (one) glyph of this box. Combining marks
            // are positioned by the font for a lower-case letter of average
            // width; over a capital or an italic they land beside it, so the
            // mark is drawn here instead, over the middle of the glyph.
            void setAccent(Layout::Accent kind, char32_t glyph) {
                accented = true;
                accent = kind;
                glyphTop = run.size * 0.25f;
                if (ImFontBaked* baked = run.font->GetFontBaked(run.size)) {
                    if (const ImFontGlyph* g = baked->FindGlyphNoFallback(static_cast<ImWchar>(glyph)))
                        glyphTop = g->Y0 * (baked->Size > 0.0f ? run.size / baked->Size : 1.0f);
                }
                asc = std::max(asc, run.ascent - glyphTop + run.size * 0.22f + run.rise);
            }

            void emit(Layout& out, float x, float baseline) const override {
                const float b = baseline - run.rise;
                const float top = b - run.ascent;
                if (run.background & 0xFF000000u)
                    out.fill(ImVec2(x, top + run.size * 0.04f), ImVec2(x + w, top + run.size * 1.04f), run.background);
                Layout::Text t;
                t.pos = ImVec2(x, top);
                t.end = x + w;
                t.baseline = b;
                t.font = run.font;
                t.size = run.size;
                t.color = run.color;
                t.italic = run.italic;
                t.text = text;
                out.text(t);
                const float pen = std::max(1.0f, std::round(run.size / 14.0f));
                if (run.overline) {
                    const float y = std::floor(b - run.size * 0.80f);
                    out.fill(ImVec2(x, y), ImVec2(x + w, y + pen), run.color);
                }
                if (run.underline) {
                    const float y = std::floor(b + run.size * 0.12f);
                    out.fill(ImVec2(x, y), ImVec2(x + w, y + pen), run.color);
                }
                if (!run.link.empty()) out.link(ImVec2(x, top), ImVec2(x + w, top + run.size), run.link);
                if (accented) {
                    Layout::AccentMark a;
                    a.kind = accent;
                    a.size = run.size;
                    a.width = w;
                    a.color = run.color;
                    a.centre = ImVec2(x + w * 0.5f, top + glyphTop - run.size * 0.11f);
                    if (run.italic) a.centre.x += kSlant * (b - run.size * kSlantPivot - a.centre.y);
                    out.accent(a);
                }
            }
        };

        struct KernBox final : Box {
            explicit KernBox(float width) { w = width; }
            void emit(Layout&, float, float) const override {}
        };

        struct HBox final : Box {
            std::vector<BoxPtr> kids;
            void add(BoxPtr b) {
                w += b->w;
                asc = std::max(asc, b->asc);
                desc = std::max(desc, b->desc);
                kids.push_back(std::move(b));
            }
            void emit(Layout& out, float x, float baseline) const override {
                for (const BoxPtr& k : kids) {
                    k->emit(out, x, baseline);
                    x += k->w;
                }
            }
        };

        struct ImageBox final : Box {
            std::shared_ptr<Texture> texture;
            ImageBox(std::shared_ptr<Texture> t, float width, float height) : texture(std::move(t)) {
                w = width;
                asc = height;
            }
            void emit(Layout& out, float x, float baseline) const override {
                out.image(ImVec2(x, baseline - asc), ImVec2(x + w, baseline), texture);
            }
        };

        // The stacked constructions of a display formula: a fraction (two
        // rows, a rule under the first) and cases / matrices (a brace that
        // spans the rows, then the rows). Columns are as wide as their widest
        // cell; the whole is centred on the math axis, a fraction with its
        // rule on it.
        struct TableBox final : Box {
            struct Cell {
                BoxPtr box;
                bool rule = false;         // a rule under the cell (the numerator)
                bool centred = false;
                bool spans = false;        // the brace: over every row, before the columns
                float padLeft = 0.0f;
            };
            std::vector<std::vector<Cell>> rows;
            float pad = 0.0f;
            float pen = 1.0f;
            float axis = 0.0f;
            ImU32 color = theme::kText;
            // computed by finish()
            std::vector<float> colW, rowAsc, rowDesc;
            float spanW = 0.0f;
            float bodyH = 0.0f;

            const Cell* spanCell() const {
                return !rows.empty() && !rows.front().empty() && rows.front().front().spans ? &rows.front().front() : nullptr;
            }

            void finish() {
                const Cell* span = spanCell();
                rowAsc.assign(rows.size(), 0.0f);
                rowDesc.assign(rows.size(), 0.0f);
                for (std::size_t r = 0; r < rows.size(); ++r) {
                    std::size_t col = 0;
                    for (const Cell& c : rows[r]) {
                        if (&c == span || !c.box) continue;
                        if (col >= colW.size()) colW.resize(col + 1, 0.0f);
                        colW[col] = std::max(colW[col], c.box->w + 2 * pad + c.padLeft);
                        rowAsc[r] = std::max(rowAsc[r], c.box->asc + pad);
                        rowDesc[r] = std::max(rowDesc[r], c.box->desc + pad + (c.rule ? pen : 0.0f));
                        ++col;
                    }
                }
                bodyH = 0.0f;
                for (std::size_t r = 0; r < rows.size(); ++r) bodyH += rowAsc[r] + rowDesc[r];
                spanW = span && span->box ? span->box->w + pad : 0.0f;
                w = spanW;
                for (float c : colW) w += c;
                const bool fraction = rows.size() == 2 && !rows[0].empty() && rows[0][0].rule;
                if (fraction) {
                    asc = rowAsc[0] + rowDesc[0] + axis;
                    desc = rowAsc[1] + rowDesc[1] - axis;
                } else {
                    asc = bodyH * 0.5f + axis;
                    desc = bodyH * 0.5f - axis;
                }
            }

            void emit(Layout& out, float x, float baseline) const override {
                const Cell* span = spanCell();
                const float top = baseline - asc;
                if (span && span->box) {
                    // centred on the rows, as tall as it comes
                    const float mid = top + bodyH * 0.5f;
                    span->box->emit(out, x, mid + (span->box->asc - span->box->desc) * 0.5f);
                }
                float y = top;
                for (std::size_t r = 0; r < rows.size(); ++r) {
                    float cx = x + spanW;
                    std::size_t col = 0;
                    for (const Cell& c : rows[r]) {
                        if (&c == span || !c.box) continue;
                        const float cw = colW[col];
                        const float room = cw - 2 * pad - c.padLeft - c.box->w;
                        const float bx = cx + pad + c.padLeft + (c.centred ? room * 0.5f : 0.0f);
                        c.box->emit(out, bx, y + rowAsc[r]);
                        if (c.rule) {
                            const float ry = std::floor(y + rowAsc[r] + rowDesc[r] - pen);
                            out.fill(ImVec2(cx, ry), ImVec2(cx + cw, ry + pen), color);
                        }
                        cx += cw;
                        ++col;
                    }
                    y += rowAsc[r] + rowDesc[r];
                }
            }
        };

        // --- inline items ---------------------------------------------------------

        struct Item {
            enum class Kind { Box,
                              Space,
                              Break };
            Kind kind = Kind::Box;
            BoxPtr box;
            Run run;               // Space: what it is drawn with
            float width = 0.0f;    // Space
        };
        using Items = std::vector<Item>;

        void addBox(Items& out, BoxPtr box) {
            Item it;
            it.kind = Item::Kind::Box;
            it.box = std::move(box);
            out.push_back(std::move(it));
        }

        void addSpace(Items& out, const Run& run, bool always) {
            // a run of spaces is one space, as in HTML
            if (!always && (out.empty() || out.back().kind != Item::Kind::Box)) return;
            Item it;
            it.kind = Item::Kind::Space;
            it.run = run;
            it.width = measure(run, " ");
            out.push_back(std::move(it));
        }

        void addBreak(Items& out) {
            Item it;
            it.kind = Item::Kind::Break;
            out.push_back(std::move(it));
        }

        // Text as words and spaces. The special spaces LaTeX's \, \; \quad
        // turn into are made of room, not of glyphs a font may lack, and a
        // combining accent becomes a mark on the letter before it.
        void addText(Items& out, const std::string& textIn, const State& st) {
            const std::string text = st.base.face == Face::Caption ? captionCase(textIn) : textIn;
            const Run run = resolve(st);
            std::string word;
            std::size_t lastStart = 0;       // where the last code point of `word` begins
            char32_t lastGlyph = 0;
            auto flush = [&] {
                if (!word.empty()) addBox(out, std::make_unique<TextBox>(word, run));
                word.clear();
                lastGlyph = 0;
            };
            bool lineStart = true;
            for (std::size_t i = 0; i < text.size();) {
                const std::size_t at = i;
                const char32_t c = nextCodepoint(text, i);
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    if (st.keepSpaces && lineStart) {
                        // indentation: room glued to the first word
                        addBox(out, std::make_unique<KernBox>(measure(run, " ") * (c == '\t' ? 4.0f : 1.0f)));
                        continue;
                    }
                    flush();
                    addSpace(out, run, st.keepSpaces);
                    continue;
                }
                lineStart = false;
                float em = 0.0f;
                switch (c) {
                    case 0x2009: em = 0.17f; break;     // thin space
                    case 0x2005: em = 0.25f; break;     // four-per-em
                    case 0x2002: em = 0.5f; break;
                    case 0x2003: em = 1.0f; break;      // \quad
                    case 0x00A0: em = -1.0f; break;     // no-break space
                    default: break;
                }
                if (em != 0.0f) {
                    flush();
                    addBox(out, std::make_unique<KernBox>(em < 0.0f ? measure(run, " ") : em * run.size));
                    continue;
                }
                Layout::Accent kind{};
                if (accentOf(c, kind)) {
                    if (lastGlyph == 0) continue;       // nothing to sit on
                    const std::string glyph = word.substr(lastStart);
                    const char32_t base = lastGlyph;
                    word.erase(lastStart);
                    flush();
                    auto box = std::make_unique<TextBox>(glyph, run);
                    box->setAccent(kind, base);
                    addBox(out, std::move(box));
                    continue;
                }
                lastStart = word.size();
                lastGlyph = c;
                word.append(text, at, i - at);
            }
            flush();
        }

        // Pieces of a word that is wider than the line, so that it can be
        // broken anywhere (a path, a URL).
        std::vector<std::string> splitToWidth(const std::string& text, const Run& run, float width) {
            std::vector<std::string> parts;
            std::string cur;
            for (std::size_t i = 0; i < text.size();) {
                const std::size_t at = i;
                nextCodepoint(text, i);
                const std::string glyph = text.substr(at, i - at);
                if (!cur.empty() && measure(run, cur + glyph) > width) {
                    parts.push_back(cur);
                    cur.clear();
                }
                cur += glyph;
            }
            if (!cur.empty()) parts.push_back(cur);
            return parts;
        }

        // Lines broken at the spaces. `width` <= 0 or FLT_MAX: one line as
        // wide as it comes.
        Layout breakLines(Items& items, const State& st, float width, Align align) {
            const Run base = resolve(st);
            const bool bounded = width > 0.0f && width < FLT_MAX * 0.5f;
            const float lineH = st.base.leading * base.size;

            // words wider than the line are cut up first
            if (bounded) {
                for (std::size_t i = 0; i < items.size(); ++i) {
                    if (items[i].kind != Item::Kind::Box) continue;
                    const auto* tb = dynamic_cast<const TextBox*>(items[i].box.get());
                    if (!tb || tb->accented || tb->w <= width) continue;
                    const bool alone = (i == 0 || items[i - 1].kind != Item::Kind::Box) &&
                                       (i + 1 == items.size() || items[i + 1].kind != Item::Kind::Box);
                    if (!alone) continue;
                    const Run run = tb->run;
                    const std::vector<std::string> parts = splitToWidth(tb->text, run, width);
                    Items pieces;
                    for (std::size_t p = 0; p < parts.size(); ++p) {
                        if (p) {
                            Item gap;
                            gap.kind = Item::Kind::Space;
                            gap.run = run;
                            gap.width = 0.0f;
                            pieces.push_back(std::move(gap));
                        }
                        addBox(pieces, std::make_unique<TextBox>(parts[p], run));
                    }
                    const std::size_t n = pieces.size();
                    items.erase(items.begin() + static_cast<std::ptrdiff_t>(i));
                    items.insert(items.begin() + static_cast<std::ptrdiff_t>(i), std::make_move_iterator(pieces.begin()),
                                 std::make_move_iterator(pieces.end()));
                    i += n - 1;
                }
            }

            Layout out;
            float y = 0.0f, widest = 0.0f;
            std::size_t i = 0;
            const std::size_t n = items.size();
            bool any = false;
            while (i < n) {
                // a line does not begin with a space
                while (i < n && items[i].kind == Item::Kind::Space && !st.keepSpaces) ++i;
                if (i >= n) break;
                const std::size_t first = i;
                std::size_t last = i;          // one past the last item on the line
                float x = 0.0f, pending = 0.0f;
                std::size_t j = i;
                bool broke = false;
                while (j < n) {
                    if (items[j].kind == Item::Kind::Break) {
                        broke = true;
                        break;
                    }
                    if (items[j].kind == Item::Kind::Space) {
                        pending += items[j].width;
                        ++j;
                        continue;
                    }
                    std::size_t k = j;
                    float group = 0.0f;
                    while (k < n && items[k].kind == Item::Kind::Box) group += items[k++].box->w;
                    if (bounded && x > 0.0f && x + pending + group > width + 0.5f) break;
                    x += pending + group;
                    pending = 0.0f;
                    last = k;
                    j = k;
                }
                // metrics of the line
                float asc = base.ascent, desc = base.size - base.ascent;
                for (std::size_t k = first; k < last; ++k) {
                    if (items[k].kind != Item::Kind::Box) continue;
                    asc = std::max(asc, items[k].box->asc);
                    desc = std::max(desc, items[k].box->desc);
                }
                const float extra = std::max(0.0f, lineH - (asc + desc));
                const float baseline = std::floor(y + extra * 0.5f + asc + 0.5f);
                float cx = align == Align::Center && bounded ? std::max(0.0f, (width - x) * 0.5f) : 0.0f;
                cx = std::floor(cx);
                for (std::size_t k = first; k < last; ++k) {
                    Item& it = items[k];
                    if (it.kind == Item::Kind::Box) {
                        it.box->emit(out, cx, baseline);
                        cx += it.box->w;
                    } else if (it.kind == Item::Kind::Space) {
                        if (it.width > 0.0f) TextBox(" ", it.run).emit(out, cx, baseline);
                        cx += it.width;
                    }
                }
                widest = std::max(widest, x);
                y += asc + desc + extra;
                any = true;
                i = broke ? j + 1 : std::max(last, first + (last == first ? 1u : 0u));
                if (!broke && last == first) i = j > first ? j : first + 1;   // nothing fitted: never stand still
            }
            out.width = bounded ? width : std::ceil(widest);
            out.height = any ? std::ceil(y) : 0.0f;
            return out;
        }

        // Everything on one line, as one box.
        BoxPtr lineBox(Items& items) {
            auto h = std::make_unique<HBox>();
            for (Item& it : items) {
                if (it.kind == Item::Kind::Box) h->add(std::move(it.box));
                else if (it.kind == Item::Kind::Space && it.run.font) h->add(std::make_unique<TextBox>(" ", it.run));
                // a line break inside one cell of a formula has no room to break into
            }
            return h;
        }

        // --- the HTML latexToHtml writes --------------------------------------------

        std::string decodeEntities(const std::string& s) {
            std::string out;
            out.reserve(s.size());
            for (std::size_t i = 0; i < s.size(); ++i) {
                if (s[i] != '&') {
                    out += s[i];
                    continue;
                }
                static const std::pair<const char*, char> entities[] = {{"&lt;", '<'}, {"&gt;", '>'}, {"&amp;", '&'}, {"&quot;", '"'}};
                bool found = false;
                for (const auto& [name, ch] : entities) {
                    const std::size_t len = std::strlen(name);
                    if (s.compare(i, len, name) == 0) {
                        out += ch;
                        i += len - 1;
                        found = true;
                        break;
                    }
                }
                if (!found) out += '&';
            }
            return out;
        }

        class MathHtml {
        public:
            explicit MathHtml(const std::string& html) : s_(html) {}

            void parse(Items& out, const State& st) {
                while (pos_ < s_.size()) inlineUntilClose(out, st);
            }

        private:
            struct Tag {
                std::string name;
                std::string attrs;
                bool closing = false;
            };

            Tag readTag() {   // at '<'
                Tag t;
                const std::size_t end = s_.find('>', pos_);
                const std::string body = s_.substr(pos_ + 1, end == std::string::npos ? std::string::npos : end - pos_ - 1);
                pos_ = end == std::string::npos ? s_.size() : end + 1;
                std::size_t i = 0;
                if (i < body.size() && body[i] == '/') {
                    t.closing = true;
                    ++i;
                }
                while (i < body.size() && std::isalnum(static_cast<unsigned char>(body[i]))) t.name += body[i++];
                t.attrs = body.substr(i);
                return t;
            }

            static float pxIn(const std::string& attrs, float otherwise) {
                const std::size_t at = attrs.find("font-size:");
                if (at == std::string::npos) return otherwise;
                const float v = static_cast<float>(std::atof(attrs.c_str() + at + 10));
                return v > 0.0f ? v : otherwise;
            }

            // Text and tags up to the tag that closes the element we are in.
            void inlineUntilClose(Items& out, const State& st) {
                while (pos_ < s_.size()) {
                    if (s_[pos_] != '<') {
                        const std::size_t next = s_.find('<', pos_);
                        const std::string text = s_.substr(pos_, next == std::string::npos ? std::string::npos : next - pos_);
                        pos_ = next == std::string::npos ? s_.size() : next;
                        addText(out, decodeEntities(text), st);
                        continue;
                    }
                    const Tag tag = readTag();
                    if (tag.closing) return;
                    State in = st;
                    if (tag.name == "br") {
                        addBreak(out);
                        continue;
                    }
                    if (tag.name == "table") {
                        addBox(out, table(st, tag.attrs));
                        continue;
                    }
                    if (tag.name == "i") {
                        in.base.italic = true;
                    } else if (tag.name == "b") {
                        in.base.weight = Weight::Bold;
                        if (tag.attrs.find("font-style: normal") != std::string::npos) in.base.italic = false;
                    } else if (tag.name == "span") {
                        if (tag.attrs.find("font-style: normal") != std::string::npos) in.base.italic = false;
                        if (tag.attrs.find("overline") != std::string::npos) in.overline = true;
                    } else if (tag.name == "sup" || tag.name == "sub") {
                        const float size = displaySize(st.base.px * st.scale);
                        in.rise += tag.name == "sup" ? kSupRise * size : -kSubDrop * size;
                        in.scale *= kScriptScale;
                    }
                    inlineUntilClose(out, in);
                }
            }

            // After <table ...>: the rows up to </table>.
            BoxPtr table(const State& st, const std::string& tableAttrs) {
                auto box = std::make_unique<TableBox>();
                const float size = displaySize(st.base.px * st.scale);
                box->pad = std::max(1.0f, std::round(size * 0.12f));
                box->pen = std::max(1.0f, std::round(size / 14.0f));
                box->axis = kAxis * size;
                box->color = st.base.color;
                (void)tableAttrs;
                bool plain = true;   // no rule, no brace: cells side by side on one baseline
                while (pos_ < s_.size()) {
                    const std::size_t next = s_.find('<', pos_);
                    if (next == std::string::npos) {
                        pos_ = s_.size();
                        break;
                    }
                    pos_ = next;
                    const Tag tag = readTag();
                    if (tag.closing) {
                        if (tag.name == "table") break;
                        continue;
                    }
                    if (tag.name == "tr") {
                        box->rows.emplace_back();
                    } else if (tag.name == "td") {
                        if (box->rows.empty()) box->rows.emplace_back();
                        TableBox::Cell cell;
                        cell.rule = tag.attrs.find("border-bottom") != std::string::npos;
                        cell.centred = tag.attrs.find("align=\"center\"") != std::string::npos;
                        cell.spans = tag.attrs.find("rowspan") != std::string::npos;
                        if (tag.attrs.find("padding-left") != std::string::npos) cell.padLeft = size * 0.5f;
                        State in = st;
                        // the brace of \begin{cases} is a glyph as tall as the rows
                        if (cell.spans) in.base.px = st.base.px * pxIn(tag.attrs, 15.0f) / 15.0f;
                        Items items;
                        inlineUntilClose(items, in);   // to </td>
                        cell.box = lineBox(items);
                        plain = plain && !cell.rule && !cell.spans;
                        box->rows.back().push_back(std::move(cell));
                    }
                }
                if (plain && box->rows.size() == 1) {
                    auto row = std::make_unique<HBox>();
                    for (TableBox::Cell& c : box->rows.front()) {
                        row->add(std::make_unique<KernBox>(box->pad));
                        row->add(std::move(c.box));
                        row->add(std::make_unique<KernBox>(box->pad));
                    }
                    return row;
                }
                box->finish();
                return box;
            }

            const std::string& s_;
            std::size_t pos_ = 0;
        };

        void addMath(Items& out, const std::string& tex, bool display, const State& st) {
            const std::string html = latexToHtml(tex, display);
            MathHtml(html).parse(out, st);
        }

        // --- inline Markdown ----------------------------------------------------------
        // The rules of core/help_pages.cpp (inlineMarkdownToHtml), in its order.

        bool isUrl(const std::string& target) {
            return target.find("://") != std::string::npos || startsWith(target, "mailto:");
        }

        std::string resolvePath(const std::string& target, const std::string& baseDir) {
            if (baseDir.empty() || isUrl(target)) return target;
            const fs::path p = fs::u8path(target);
            if (p.is_absolute()) return target;
            return (fs::u8path(baseDir) / p).lexically_normal().u8string();
        }

        struct InlineParser {
            const Context& context;
            float maxWidth;        // images are fitted to it

            void image(Items& out, const std::string& label, const std::string& target, const State& st) const {
                const std::string path = resolvePath(target, context.baseDir);
                std::shared_ptr<Texture> tex = context.images ? context.images->get(path) : nullptr;
                if (!tex || tex->width() <= 0) {
                    // what a browser shows for an image it cannot load
                    State alt = st;
                    alt.base.color = theme::kNeutral600;
                    alt.base.italic = true;
                    addText(out, label.empty() ? fileName(path) : label, alt);
                    return;
                }
                float w = px(static_cast<float>(tex->width())), h = px(static_cast<float>(tex->height()));
                if (maxWidth > 0.0f && maxWidth < FLT_MAX * 0.5f && w > maxWidth) {
                    h *= maxWidth / w;
                    w = maxWidth;
                }
                addBox(out, std::make_unique<ImageBox>(tex, std::floor(w), std::floor(h)));
            }

            // s[i] == '[': "[label](target)". Leaves i on the ')'.
            bool linkOrImage(const std::string& s, std::size_t& i, bool isImage, Items& out, const State& st) const {
                const std::size_t close = s.find(']', i);
                if (close == std::string::npos || close + 1 >= s.size() || s[close + 1] != '(') return false;
                const std::size_t end = s.find(')', close + 2);
                if (end == std::string::npos) return false;
                const std::string label = s.substr(i + 1, close - i - 1);
                const std::string target = s.substr(close + 2, end - close - 2);
                i = end;
                if (isImage) {
                    image(out, label, target, st);
                    return true;
                }
                State in = st;
                in.link = resolvePath(target, context.baseDir);
                in.base.color = theme::kAccentText;
                in.underline = true;
                parse(label, in, out);
                return true;
            }

            void parse(const std::string& s, const State& st, Items& out) const {
                std::string text;
                auto flush = [&] {
                    if (!text.empty()) addText(out, text, st);
                    text.clear();
                };
                for (std::size_t i = 0; i < s.size(); ++i) {
                    const char c = s[i];
                    if (c == '\\' && i + 1 < s.size() && (s[i + 1] == '$' || s[i + 1] == '*' || s[i + 1] == '`' || s[i + 1] == '|')) {
                        text += s[++i];
                        continue;
                    }
                    if (c == '$') {
                        const bool dbl = i + 1 < s.size() && s[i + 1] == '$';
                        const std::size_t fence = dbl ? 2 : 1;
                        const std::size_t end = s.find(dbl ? "$$" : "$", i + fence);
                        if (end != std::string::npos) {
                            flush();
                            addMath(out, s.substr(i + fence, end - i - fence), false, st);
                            i = end + fence - 1;
                            continue;
                        }
                    }
                    if (c == '`') {
                        const std::size_t end = s.find('`', i + 1);
                        if (end != std::string::npos) {
                            flush();
                            State in = st;
                            in.base.face = Face::Mono;
                            in.base.italic = false;
                            in.base.px = st.base.px * 0.94f;   // the monospace face runs large
                            in.code = true;
                            addText(out, s.substr(i + 1, end - i - 1), in);
                            i = end;
                            continue;
                        }
                    }
                    if (c == '*' && i + 1 < s.size() && s[i + 1] == '*') {
                        const std::size_t end = s.find("**", i + 2);
                        if (end != std::string::npos) {
                            flush();
                            State in = st;
                            in.base.weight = st.base.weight >= Weight::Bold ? Weight::ExtraBold : Weight::Bold;
                            parse(s.substr(i + 2, end - i - 2), in, out);
                            i = end + 1;
                            continue;
                        }
                    }
                    if (c == '*') {
                        const std::size_t end = s.find('*', i + 1);
                        if (end != std::string::npos && end > i + 1) {
                            flush();
                            State in = st;
                            in.base.italic = true;
                            parse(s.substr(i + 1, end - i - 1), in, out);
                            i = end;
                            continue;
                        }
                    }
                    if (c == '!' && i + 1 < s.size() && s[i + 1] == '[') {
                        std::size_t j = i + 1;
                        Items made;
                        if (linkOrImage(s, j, true, made, st)) {
                            flush();
                            for (Item& it : made) out.push_back(std::move(it));
                            i = j;
                            continue;
                        }
                    }
                    if (c == '[') {
                        std::size_t j = i;
                        Items made;
                        if (linkOrImage(s, j, false, made, st)) {
                            flush();
                            for (Item& it : made) out.push_back(std::move(it));
                            i = j;
                            continue;
                        }
                    }
                    if (c == '<') {
                        // the few tags the pages use; anything else is text
                        bool taken = false;
                        for (const char* br : {"<br>", "<br/>", "<br />"}) {
                            const std::size_t len = std::strlen(br);
                            if (s.compare(i, len, br) == 0) {
                                flush();
                                addBreak(out);
                                i += len - 1;
                                taken = true;
                                break;
                            }
                        }
                        if (taken) continue;
                        for (const char* name : {"sub", "sup", "b", "i"}) {
                            const std::string open = std::string("<") + name + ">", close = std::string("</") + name + ">";
                            if (s.compare(i, open.size(), open) == 0) {
                                flush();
                                const std::size_t end = s.find(close, i + open.size());
                                const std::string inner =
                                    s.substr(i + open.size(), end == std::string::npos ? std::string::npos : end - i - open.size());
                                State in = st;
                                if (name[0] == 'b') {
                                    in.base.weight = Weight::Bold;
                                } else if (name[0] == 'i') {
                                    in.base.italic = true;
                                } else {
                                    const float size = displaySize(st.base.px * st.scale);
                                    in.rise += name[2] == 'p' ? kSupRise * size : -kSubDrop * size;
                                    in.scale *= kScriptScale;
                                }
                                parse(inner, in, out);
                                i = end == std::string::npos ? s.size() : end + close.size() - 1;
                                taken = true;
                                break;
                            }
                            if (s.compare(i, close.size(), close) == 0) {   // a closing tag on its own
                                i += close.size() - 1;
                                taken = true;
                                break;
                            }
                        }
                        if (taken) continue;
                    }
                    text += c;
                }
                flush();
            }
        };

        State stateOf(const TextStyle& style) {
            State st;
            st.base = style;
            return st;
        }

        // --- blocks -------------------------------------------------------------------

        std::string trim(const std::string& s) { return trimmed(s); }

        std::vector<std::string> splitLines(const std::string& text) {
            std::vector<std::string> lines;
            std::string cur;
            for (char c : text) {
                if (c == '\n') {
                    if (!cur.empty() && cur.back() == '\r') cur.pop_back();
                    lines.push_back(cur);
                    cur.clear();
                } else {
                    cur += c;
                }
            }
            if (!cur.empty()) lines.push_back(cur);
            return lines;
        }

        // Split a Markdown table row on '|' outside $...$ math and not escaped.
        std::vector<std::string> splitTableRow(const std::string& line) {
            std::vector<std::string> cells;
            std::string cur;
            bool inMath = false, inCode = false;
            for (std::size_t i = 0; i < line.size(); ++i) {
                const char c = line[i];
                if (c == '\\' && i + 1 < line.size()) {
                    cur += c;
                    cur += line[++i];
                    continue;
                }
                if (c == '`' && !inMath) inCode = !inCode;
                if (c == '$' && !inCode) inMath = !inMath;
                if (c == '|' && !inMath && !inCode) {
                    cells.push_back(cur);
                    cur.clear();
                } else {
                    cur += c;
                }
            }
            cells.push_back(cur);
            if (!cells.empty() && trim(cells.front()).empty()) cells.erase(cells.begin());
            if (!cells.empty() && trim(cells.back()).empty()) cells.pop_back();
            for (std::string& c : cells) c = trim(c);
            return cells;
        }

        bool isTableSeparator(const std::string& line) {
            const std::string t = trim(line);
            if (t.empty() || t[0] != '|') return false;
            return t.find_first_not_of("|-: \t") == std::string::npos;
        }

        bool isBullet(const std::string& t) { return startsWith(t, "- ") || startsWith(t, "* "); }

        bool isNumbered(const std::string& t) {
            return t.size() > 2 && std::isdigit(static_cast<unsigned char>(t[0])) && t.find(". ") != std::string::npos && t.find(". ") < 4;
        }

        bool isFence(const std::string& t) { return startsWith(t, "```") || startsWith(t, "~~~"); }

        std::size_t indentOf(const std::string& line) {
            std::size_t n = 0;
            while (n < line.size() && (line[n] == ' ' || line[n] == '\t')) ++n;
            return n;
        }

        // A display formula starting at lines[i] (which begins with "$$").
        std::string readDisplayBlock(const std::vector<std::string>& lines, std::size_t& i) {
            std::string first = trim(lines[i]);
            first.erase(0, 2);
            const std::size_t closeSame = first.find("$$");
            if (closeSame != std::string::npos) {
                ++i;
                return trim(first.substr(0, closeSame));
            }
            std::string tex = first;
            ++i;
            while (i < lines.size()) {
                const std::string t = trim(lines[i]);
                const std::size_t close = t.find("$$");
                if (close != std::string::npos) {
                    tex += "\n" + t.substr(0, close);
                    ++i;
                    break;
                }
                tex += "\n" + lines[i];
                ++i;
            }
            return trim(tex);
        }

        // Blocks stacked with collapsing margins: the gap between two blocks
        // is the larger of what the upper one wants below and the lower one
        // above; the first block starts at the top.
        struct Column {
            Layout out;
            float y = 0.0f;
            float below = 0.0f;
            bool first = true;

            void add(const Layout& part, float above, float under, float x = 0.0f) {
                if (!first) y += std::max(below, above);
                first = false;
                out.place(part, x, std::floor(y));
                y += part.height;
                below = under;
            }
        };

        Layout tableBlock(const std::vector<std::string>& rows, const TextStyle& body, float width, const Context& context) {
            std::vector<std::vector<std::string>> cells;
            for (const std::string& line : rows)
                if (!isTableSeparator(line)) cells.push_back(splitTableRow(line));
            std::size_t columns = 0;
            for (const auto& r : cells) columns = std::max(columns, r.size());
            Layout out;
            out.width = width;
            if (columns == 0) return out;
            const float pad = px(6);
            // the first column is the narrow one that names the row
            std::vector<float> widths(columns, 0.0f);
            if (columns == 1) {
                widths[0] = width;
            } else {
                widths[0] = std::floor(std::min(px(130) + 2 * pad, width * 0.4f));
                const float rest = std::floor((width - widths[0]) / static_cast<float>(columns - 1));
                for (std::size_t c = 1; c < columns; ++c) widths[c] = rest;
            }
            float y = 0.0f;
            for (std::size_t r = 0; r < cells.size(); ++r) {
                const bool header = r == 0;
                float x = 0.0f, rowH = 0.0f;
                std::vector<Layout> parts;
                for (std::size_t c = 0; c < columns; ++c) {
                    TextStyle s = body;
                    if (header) {
                        s.px = theme::kCaptionPx;
                        s.face = Face::Caption;
                        s.color = theme::kNeutral600;
                        s.weight = Weight::Regular;
                    } else if (c == 0) {
                        s.weight = Weight::Bold;
                    }
                    const std::string text = c < cells[r].size() ? cells[r][c] : std::string();
                    parts.push_back(paragraph(text, s, std::max(px(20), widths[c] - 2 * pad), context));
                    rowH = std::max(rowH, parts.back().height);
                }
                for (std::size_t c = 0; c < columns; ++c) {
                    out.place(parts[c], x + pad, y + pad);
                    x += widths[c];
                }
                y += rowH + 2 * pad;
                const float pen = theme::crispPen(header ? 2.0f : 1.0f);
                out.fill(ImVec2(0.0f, std::floor(y)), ImVec2(width, std::floor(y) + pen), theme::kDivider);
                y = std::floor(y) + pen;
            }
            out.width = width;
            out.height = y;
            return out;
        }

        Layout codeBlock(const std::vector<std::string>& lines, const TextStyle& body, float width) {
            const float padX = px(10), padY = px(8);
            State st;
            st.base = body;
            st.base.face = Face::Mono;
            st.base.px = 12;
            st.base.weight = Weight::Regular;
            st.base.italic = false;
            st.base.color = theme::kNeutral800;
            st.base.leading = 1.3f;
            st.keepSpaces = true;
            Layout out;
            float y = padY;
            for (const std::string& line : lines) {
                Items items;
                addText(items, line.empty() ? std::string(" ") : line, st);
                const Layout l = breakLines(items, st, std::max(px(40), width - 2 * padX), Align::Left);
                const float h = l.height > 0.0f ? l.height : st.base.leading * displaySize(st.base.px);
                out.place(l, padX, y);
                y += h;
            }
            Layout framed;
            framed.fill(ImVec2(0, 0), ImVec2(width, y + padY), theme::kSurface);
            framed.place(out, 0.0f, 0.0f);
            framed.width = width;
            framed.height = y + padY;
            return framed;
        }

        Layout displayBlock(const std::string& tex, const TextStyle& body, float width) {
            const float pad = px(12);
            TextStyle s = body;
            s.px = theme::kMonoPx;   // 15 px, the design's size for formulas
            const Layout f = formula(tex, true, s, std::max(px(40), width - 2 * pad));
            Layout out;
            out.fill(ImVec2(0, 0), ImVec2(width, f.height + 2 * pad), theme::kSurface);
            out.frame(ImVec2(0, 0), ImVec2(width, f.height + 2 * pad), theme::kDivider, 1.0f);
            out.place(f, pad, pad);
            out.width = width;
            out.height = f.height + 2 * pad;
            return out;
        }

        Layout listBlock(const std::vector<std::pair<std::string, std::string>>& entries, const TextStyle& body, float width,
                         const Context& context) {
            // marker column: wide enough for "12."
            float markerW = px(18);
            for (const auto& e : entries) markerW = std::max(markerW, theme::textSize(e.first, body.px).x + px(8));
            Layout out;
            float y = 0.0f;
            for (std::size_t k = 0; k < entries.size(); ++k) {
                if (k) y += px(3);
                TextStyle m = body;
                m.color = theme::kNeutral700;
                const Layout marker = plain(entries[k].first, m, 0.0f);
                const Layout text = paragraph(entries[k].second, body, std::max(px(40), width - markerW), context);
                out.place(marker, std::max(0.0f, markerW - px(6) - marker.width), y);
                out.place(text, markerW, y);
                y += std::max(marker.height, text.height);
            }
            out.width = width;
            out.height = y;
            return out;
        }

    } // namespace

    // --- Images ------------------------------------------------------------------------

    std::shared_ptr<Texture> Images::get(const std::string& path) {
        if (auto it = textures_.find(path); it != textures_.end()) return it->second;
        std::shared_ptr<Texture> tex;
        std::vector<std::uint8_t> rgba;
        int w = 0, h = 0;
        if (isFile(path) && readImage(path, rgba, w, h) && w > 0 && h > 0) {
            tex = std::make_shared<Texture>();
            tex->upload(rgba.data(), w, h, true);
        }
        textures_[path] = tex;   // a file that cannot be read is not tried again every frame
        return tex;
    }

    void Images::clear() { textures_.clear(); }

    // --- Layout ------------------------------------------------------------------------

    void Layout::fill(ImVec2 min, ImVec2 max, ImU32 color) {
        Rect r;
        r.min = min;
        r.max = max;
        r.color = color;
        rects_.push_back(r);
        width = std::max(width, max.x);
        height = std::max(height, max.y);
    }

    void Layout::frame(ImVec2 min, ImVec2 max, ImU32 color, float designPx, bool dashed) {
        Rect r;
        r.min = min;
        r.max = max;
        r.color = color;
        r.pen = designPx;
        r.dashed = dashed;
        rects_.push_back(r);
        width = std::max(width, max.x);
        height = std::max(height, max.y);
    }

    void Layout::image(ImVec2 min, ImVec2 max, std::shared_ptr<Texture> texture) {
        Picture p;
        p.min = min;
        p.max = max;
        p.texture = std::move(texture);
        images_.push_back(std::move(p));
    }

    void Layout::link(ImVec2 min, ImVec2 max, const std::string& url) {
        // the words of one link on one line are one area
        if (!links_.empty()) {
            Link& last = links_.back();
            if (last.url == url && std::abs(last.min.y - min.y) < 0.5f && std::abs(last.max.x - min.x) < 0.5f) {
                last.max.x = max.x;
                return;
            }
        }
        Link l;
        l.min = min;
        l.max = max;
        l.url = url;
        links_.push_back(std::move(l));
    }

    void Layout::text(const Text& t) {
        if (t.text.empty()) return;
        // Words that follow each other in one style are one string: the
        // glyphs then advance as the font says, where words placed one by one
        // would each start on a whole pixel.
        if (!texts_.empty()) {
            Text& last = texts_.back();
            if (last.font == t.font && last.size == t.size && last.color == t.color && last.italic == t.italic &&
                std::abs(last.baseline - t.baseline) < 0.01f && std::abs(last.pos.y - t.pos.y) < 0.01f &&
                std::abs(last.end - t.pos.x) < 0.01f) {
                last.text += t.text;
                last.end = t.end;
                return;
            }
        }
        texts_.push_back(t);
    }

    void Layout::accent(const AccentMark& a) { accents_.push_back(a); }

    void Layout::place(const Layout& part, float x, float y) {
        const ImVec2 d(x, y);
        for (Rect r : part.rects_) {
            r.min = ImVec2(r.min.x + d.x, r.min.y + d.y);
            r.max = ImVec2(r.max.x + d.x, r.max.y + d.y);
            rects_.push_back(r);
        }
        for (Text t : part.texts_) {
            t.pos = ImVec2(t.pos.x + d.x, t.pos.y + d.y);
            t.end += d.x;
            t.baseline += d.y;
            texts_.push_back(std::move(t));
        }
        for (AccentMark a : part.accents_) {
            a.centre = ImVec2(a.centre.x + d.x, a.centre.y + d.y);
            accents_.push_back(a);
        }
        for (Picture p : part.images_) {
            p.min = ImVec2(p.min.x + d.x, p.min.y + d.y);
            p.max = ImVec2(p.max.x + d.x, p.max.y + d.y);
            images_.push_back(std::move(p));
        }
        for (Link l : part.links_) {
            l.min = ImVec2(l.min.x + d.x, l.min.y + d.y);
            l.max = ImVec2(l.max.x + d.x, l.max.y + d.y);
            links_.push_back(std::move(l));
        }
        width = std::max(width, x + part.width);
        height = std::max(height, y + part.height);
    }

    void Layout::draw(ImDrawList* dl, ImVec2 originIn) const {
        const ImVec2 o(theme::snap(originIn.x), theme::snap(originIn.y));
        const float clipTop = dl->GetClipRectMin().y, clipBottom = dl->GetClipRectMax().y;
        auto hidden = [&](float top, float bottom) { return o.y + bottom < clipTop || o.y + top > clipBottom; };

        for (const Rect& r : rects_) {
            if (hidden(r.min.y, r.max.y)) continue;
            const ImVec2 a(o.x + r.min.x, o.y + r.min.y), b(o.x + r.max.x, o.y + r.max.y);
            if (r.pen <= 0.0f) dl->AddRectFilled(ImVec2(theme::snap(a.x), theme::snap(a.y)), ImVec2(theme::snap(b.x), theme::snap(b.y)), r.color);
            else if (r.dashed) widgets::dashedRect(dl, a, b, r.color, r.pen);
            else widgets::crispRect(dl, a, b, r.color, r.pen);
        }
        for (const Picture& p : images_) {
            if (!p.texture || !p.texture->valid() || hidden(p.min.y, p.max.y)) continue;
            dl->AddImage(p.texture->ref(), ImVec2(theme::snap(o.x + p.min.x), theme::snap(o.y + p.min.y)),
                         ImVec2(theme::snap(o.x + p.max.x), theme::snap(o.y + p.max.y)));
        }
        for (const Text& t : texts_) {
            if (hidden(t.pos.y - t.size, t.pos.y + 2 * t.size)) continue;
            const int first = dl->VtxBuffer.Size;
            dl->AddText(t.font, t.size, ImVec2(o.x + t.pos.x, o.y + t.pos.y), t.color, t.text.c_str(), t.text.c_str() + t.text.size());
            if (t.italic) {
                const float pivot = std::floor(o.y + t.baseline) - t.size * kSlantPivot;
                for (int i = first; i < dl->VtxBuffer.Size; ++i) {
                    ImDrawVert& v = dl->VtxBuffer.Data[i];
                    v.pos.x += kSlant * (pivot - v.pos.y);
                }
            }
        }
        for (const AccentMark& a : accents_) {
            if (hidden(a.centre.y - a.size, a.centre.y + a.size)) continue;
            const ImVec2 c(o.x + a.centre.x, o.y + a.centre.y);
            const float pen = std::max(1.0f, a.size / 13.0f);
            const float half = std::clamp(a.width * 0.42f, a.size * 0.16f, a.size * 0.3f);
            const float rise = a.size * 0.065f;
            switch (a.kind) {
                case Accent::Tilde: {
                    ImVec2 pts[13];
                    for (int i = 0; i < 13; ++i) {
                        const float u = static_cast<float>(i) / 12.0f;
                        pts[i] = ImVec2(c.x - half + 2 * half * u, c.y - rise * std::sin(u * 6.2831853f));
                    }
                    dl->AddPolyline(pts, 13, a.color, ImDrawFlags_None, pen);
                    break;
                }
                case Accent::Hat: {
                    const ImVec2 pts[3] = {ImVec2(c.x - half * 0.8f, c.y + rise), ImVec2(c.x, c.y - rise), ImVec2(c.x + half * 0.8f, c.y + rise)};
                    dl->AddPolyline(pts, 3, a.color, ImDrawFlags_None, pen);
                    break;
                }
                case Accent::Bar: dl->AddLine(ImVec2(c.x - half, c.y), ImVec2(c.x + half, c.y), a.color, pen); break;
                case Accent::Vec: {
                    dl->AddLine(ImVec2(c.x - half, c.y), ImVec2(c.x + half, c.y), a.color, pen);
                    const ImVec2 head[3] = {ImVec2(c.x + half - rise * 1.4f, c.y - rise), ImVec2(c.x + half, c.y),
                                            ImVec2(c.x + half - rise * 1.4f, c.y + rise)};
                    dl->AddPolyline(head, 3, a.color, ImDrawFlags_None, pen);
                    break;
                }
                case Accent::Dot: dl->AddCircleFilled(c, std::max(1.0f, a.size * 0.055f), a.color); break;
                case Accent::DDot:
                    dl->AddCircleFilled(ImVec2(c.x - a.size * 0.1f, c.y), std::max(1.0f, a.size * 0.055f), a.color);
                    dl->AddCircleFilled(ImVec2(c.x + a.size * 0.1f, c.y), std::max(1.0f, a.size * 0.055f), a.color);
                    break;
            }
        }
    }

    const std::string* Layout::linkAt(ImVec2 origin, ImVec2 mouse) const {
        const ImVec2 m(mouse.x - theme::snap(origin.x), mouse.y - theme::snap(origin.y));
        for (const Link& l : links_)
            if (m.x >= l.min.x && m.x < l.max.x && m.y >= l.min.y && m.y < l.max.y) return &l.url;
        return nullptr;
    }

    // --- entry points ------------------------------------------------------------------

    Layout paragraph(const std::string& markdown, const TextStyle& style, float width, const Context& context, Align align) {
        const State st = stateOf(style);
        Items items;
        const InlineParser parser{context, width};
        parser.parse(markdown, st, items);
        return breakLines(items, st, width, align);
    }

    Layout plain(const std::string& text, const TextStyle& style, float width, Align align) {
        const State st = stateOf(style);
        Items items;
        addText(items, text, st);
        return breakLines(items, st, width, align);
    }

    Layout formula(const std::string& tex, bool display, const TextStyle& style, float width) {
        State st = stateOf(style);
        if (!display) {
            Items items;
            addMath(items, tex, false, st);
            return breakLines(items, st, width, Align::Left);
        }
        // One line, however wide; made smaller until it fits. A formula is
        // not text: broken in two it would read as two formulas.
        st.base.leading = 1.0f;
        Layout out;
        for (int attempt = 0; attempt < 3; ++attempt) {
            Items items;
            addMath(items, tex, true, st);
            out = breakLines(items, st, 0.0f, Align::Left);
            if (width <= 0.0f || out.width <= width || st.base.px <= 8.0f) break;
            st.base.px = std::max(8.0f, std::floor(st.base.px * width / out.width * 10.0f) / 10.0f - (attempt ? 0.3f : 0.0f));
        }
        return out;
    }

    Layout blocks(const std::string& markdownIn, const TextStyle& body, float width, const Context& context) {
        const std::string markdown = normalizeMathDelimiters(markdownIn);
        std::vector<std::string> lines = splitLines(markdown);
        // front matter
        if (!lines.empty() && trim(lines[0]) == "---") {
            std::size_t end = 1;
            while (end < lines.size() && trim(lines[end]) != "---") ++end;
            lines.erase(lines.begin(), lines.begin() + static_cast<std::ptrdiff_t>(std::min(end + 1, lines.size())));
        }

        Column col;
        std::size_t i = 0;
        while (i < lines.size()) {
            const std::string t = trim(lines[i]);
            if (t.empty()) {
                ++i;
                continue;
            }
            if (isFence(t)) {
                const std::string fence = t.substr(0, 3);
                std::vector<std::string> code;
                ++i;
                while (i < lines.size() && !startsWith(trim(lines[i]), fence)) code.push_back(lines[i++]);
                if (i < lines.size()) ++i;   // the closing fence
                col.add(codeBlock(code, body, width), px(8), px(8));
                continue;
            }
            if (startsWith(t, "$$")) {
                col.add(displayBlock(readDisplayBlock(lines, i), body, width), px(8), px(8));
                continue;
            }
            if (t[0] == '#') {
                std::size_t level = 0;
                while (level < t.size() && t[level] == '#') ++level;
                TextStyle s = body;
                s.px = level <= 1 ? 20.0f : level == 2 ? 15.0f
                                                       : 13.0f;
                s.weight = Weight::ExtraBold;
                s.leading = 1.25f;
                col.add(paragraph(trim(t.substr(level)), s, width, context), px(14), px(4));
                ++i;
                continue;
            }
            if (t[0] == '|') {
                std::vector<std::string> rows;
                while (i < lines.size() && !trim(lines[i]).empty() && trim(lines[i])[0] == '|') rows.push_back(lines[i++]);
                col.add(tableBlock(rows, body, width, context), px(6), px(8));
                continue;
            }
            if (isBullet(t) || isNumbered(t)) {
                std::vector<std::pair<std::string, std::string>> entries;   // marker, text
                while (i < lines.size()) {
                    const std::string it = trim(lines[i]);
                    const bool bullet = isBullet(it), numbered = isNumbered(it);
                    if (!(bullet || numbered)) {
                        // an indented line continues the entry above it
                        if (!it.empty() && !entries.empty() && indentOf(lines[i]) >= 2 && it[0] != '#' && it[0] != '|' &&
                            !startsWith(it, "$$") && !isFence(it)) {
                            entries.back().second += " " + it;
                            ++i;
                            continue;
                        }
                        break;
                    }
                    if (bullet) entries.emplace_back("\xE2\x80\xA2", it.substr(2));
                    else entries.emplace_back(it.substr(0, it.find(". ") + 1), it.substr(it.find(". ") + 2));
                    ++i;
                }
                col.add(listBlock(entries, body, width, context), px(8), px(8));
                continue;
            }
            std::string para;
            while (i < lines.size()) {
                const std::string pt = trim(lines[i]);
                if (pt.empty() || pt[0] == '#' || pt[0] == '|' || startsWith(pt, "$$") || isBullet(pt) || isFence(pt)) break;
                if (!para.empty()) para += " ";
                para += pt;
                ++i;
            }
            col.add(paragraph(para, body, width, context), px(6), px(6));
        }
        col.out.width = width;
        col.out.height = std::ceil(col.y);
        return col.out;
    }

    bool show(const Layout& layout) {
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(layout.width, layout.height));
        if (!ImGui::IsItemVisible()) return false;
        layout.draw(ImGui::GetWindowDrawList(), origin);
        if (!ImGui::IsItemHovered()) return false;
        const std::string* url = layout.linkAt(origin, ImGui::GetIO().MousePos);
        if (!url) return false;
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        widgets::tooltip(*url);
        if (!ImGui::IsMouseReleased(ImGuiMouseButton_Left)) return false;
        if (isUrl(*url)) platform::openUrl(*url);
        else platform::openInFileManager(*url);
        return true;
    }

} // namespace sirius::app::gui::markdown
