#include "imgui/panels/assistant_markdown.hpp"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdlib>

#include "core/help_pages.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"

namespace sirius::app::gui::assistant_markdown {

    namespace {

        using theme::px;

        // --- parsing ----------------------------------------------------------

        struct Style {
            bool bold = false;
            bool italic = false;
            bool code = false;
            int script = 0;
            std::string href;
        };

        bool sameStyle(const Span& s, const Style& st) {
            return !s.lineBreak && s.bold == st.bold && s.italic == st.italic && s.code == st.code && s.script == st.script &&
                   s.href == st.href;
        }

        void push(std::vector<Span>& out, const std::string& text, const Style& st) {
            if (text.empty()) return;
            if (!out.empty() && sameStyle(out.back(), st)) {
                out.back().text += text;
                return;
            }
            Span s;
            s.text = text;
            s.bold = st.bold;
            s.italic = st.italic;
            s.code = st.code;
            s.script = st.script;
            s.href = st.href;
            out.push_back(std::move(s));
        }

        void pushBreak(std::vector<Span>& out) {
            Span s;
            s.lineBreak = true;
            out.push_back(std::move(s));
        }

        // "&amp;" at s[i] -> "&", with i moved to the ';'. Unknown: the '&' itself.
        std::string entity(const std::string& s, std::size_t& i) {
            const std::size_t end = s.find(';', i);
            if (end == std::string::npos || end - i > 9) return "&";
            const std::string name = s.substr(i + 1, end - i - 1);
            std::string out;
            if (name == "amp") out = "&";
            else if (name == "lt") out = "<";
            else if (name == "gt") out = ">";
            else if (name == "quot") out = "\"";
            else if (name == "apos") out = "'";
            else if (name == "nbsp") out = "\xC2\xA0";
            else if (name.size() > 1 && name[0] == '#') {
                const bool hex = name[1] == 'x' || name[1] == 'X';
                char* stop = nullptr;
                const unsigned long code = std::strtoul(name.c_str() + (hex ? 2 : 1), &stop, hex ? 16 : 10);
                if (!stop || *stop != '\0' || code == 0 || code > 0x10FFFF) return "&";
                appendUtf8(out, static_cast<char32_t>(code));
            } else {
                return "&";
            }
            i = end;
            return out;
        }

        // What latexToHtml writes (text, <sub>, <sup>, <span>, <i>, <b>, <br>,
        // entities) as spans. Tables do not occur: formulas are asked for inline.
        void appendHtml(const std::string& html, const Style& base, std::vector<Span>& out) {
            std::vector<Style> stack{base};
            std::string text;
            auto flush = [&] {
                push(out, text, stack.back());
                text.clear();
            };
            for (std::size_t i = 0; i < html.size(); ++i) {
                const char c = html[i];
                if (c == '<') {
                    const std::size_t end = html.find('>', i);
                    if (end == std::string::npos) {
                        text += c;
                        continue;
                    }
                    std::string tag = toLower(html.substr(i + 1, end - i - 1));
                    i = end;
                    const bool closing = !tag.empty() && tag[0] == '/';
                    if (closing) tag.erase(0, 1);
                    const std::size_t nameEnd = tag.find_first_of(" \t/");
                    const std::string name = nameEnd == std::string::npos ? tag : tag.substr(0, nameEnd);
                    if (name == "br") {
                        flush();
                        pushBreak(out);
                        continue;
                    }
                    if (name != "sub" && name != "sup" && name != "b" && name != "i" && name != "span") continue;
                    flush();
                    if (closing) {
                        if (stack.size() > 1) stack.pop_back();
                        continue;
                    }
                    Style st = stack.back();
                    if (name == "sub") st.script = -1;
                    else if (name == "sup") st.script = 1;
                    else if (name == "b") st.bold = true;
                    else if (name == "i") st.italic = true;
                    stack.push_back(st);
                    continue;
                }
                if (c == '&') {
                    text += entity(html, i);
                    continue;
                }
                text += c;
            }
            flush();
        }

        void parseInline(const std::string& s, const Style& st, std::vector<Span>& out);

        // s[i] == '[': "[label](target)". False when it is not a link.
        bool linkAt(const std::string& s, std::size_t& i, const Style& st, std::vector<Span>& out, bool image) {
            const std::size_t close = s.find(']', i);
            if (close == std::string::npos || close + 1 >= s.size() || s[close + 1] != '(') return false;
            const std::size_t end = s.find(')', close + 2);
            if (end == std::string::npos) return false;
            const std::string label = s.substr(i + 1, close - i - 1);
            const std::string target = trimmed(s.substr(close + 2, end - close - 2));
            i = end;
            if (image) {
                // no pictures in the transcript: what it would have shown, by name
                push(out, label.empty() ? target : label, st);
                return true;
            }
            Style link = st;
            link.href = target;
            parseInline(label.empty() ? target : label, link, out);
            return true;
        }

        // The inline subset of core/help_pages (inlineMarkdownToHtml), in its order.
        void parseInline(const std::string& s, const Style& st, std::vector<Span>& out) {
            std::string text;
            auto flush = [&] {
                push(out, text, st);
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
                        appendHtml(latexToHtml(s.substr(i + fence, end - i - fence), false), st, out);
                        i = end + fence - 1;
                        continue;
                    }
                }
                if (c == '`') {
                    const std::size_t end = s.find('`', i + 1);
                    if (end != std::string::npos) {
                        flush();
                        Style code = st;
                        code.code = true;
                        push(out, s.substr(i + 1, end - i - 1), code);
                        i = end;
                        continue;
                    }
                }
                if (c == '*' && i + 1 < s.size() && s[i + 1] == '*') {
                    const std::size_t end = s.find("**", i + 2);
                    if (end != std::string::npos) {
                        flush();
                        Style bold = st;
                        bold.bold = true;
                        parseInline(s.substr(i + 2, end - i - 2), bold, out);
                        i = end + 1;
                        continue;
                    }
                }
                if (c == '*') {
                    const std::size_t end = s.find('*', i + 1);
                    if (end != std::string::npos && end > i + 1) {
                        flush();
                        Style italic = st;
                        italic.italic = true;
                        parseInline(s.substr(i + 1, end - i - 1), italic, out);
                        i = end;
                        continue;
                    }
                }
                if (c == '!' && i + 1 < s.size() && s[i + 1] == '[') {
                    std::size_t j = i + 1;
                    flush();
                    if (linkAt(s, j, st, out, true)) {
                        i = j;
                        continue;
                    }
                }
                if (c == '[') {
                    std::size_t j = i;
                    flush();
                    if (linkAt(s, j, st, out, false)) {
                        i = j;
                        continue;
                    }
                }
                if (c == '<') {
                    static const char* allowed[] = {"<br>", "<br/>", "<br />", "<sub>", "</sub>", "<sup>", "</sup>", "<b>", "</b>", "<i>", "</i>"};
                    bool passed = false;
                    for (const char* tag : allowed) {
                        const std::size_t n = std::char_traits<char>::length(tag);
                        if (s.compare(i, n, tag) != 0) continue;
                        if (tag[1] == 'b' && tag[2] == 'r') {
                            flush();
                            pushBreak(out);
                        }
                        // the other tags would need their closing partner found;
                        // they are rare in an answer and are dropped, their text kept
                        i += n - 1;
                        passed = true;
                        break;
                    }
                    if (passed) continue;
                }
                text += c;
            }
            flush();
        }

        std::vector<std::string> splitLines(const std::string& s) {
            std::vector<std::string> lines = split(s, '\n');
            for (std::string& l : lines)
                if (!l.empty() && l.back() == '\r') l.pop_back();
            return lines;
        }

        bool isBullet(const std::string& t) { return startsWith(t, "- ") || startsWith(t, "* ") || startsWith(t, "+ "); }

        // "12. text": where the text starts, 0 when it is not a numbered item.
        std::size_t numberedAt(const std::string& t) {
            std::size_t i = 0;
            while (i < t.size() && i < 3 && std::isdigit(static_cast<unsigned char>(t[i]))) ++i;
            if (i == 0 || i + 1 >= t.size()) return 0;
            if ((t[i] != '.' && t[i] != ')') || t[i + 1] != ' ') return 0;
            return i + 2;
        }

        bool isRule(const std::string& t) {
            if (t.size() < 3) return false;
            const char c = t[0];
            if (c != '-' && c != '*' && c != '_') return false;
            return std::all_of(t.begin(), t.end(), [c](char x) { return x == c || x == ' '; }) &&
                   std::count(t.begin(), t.end(), c) >= 3;
        }

        bool isTableSeparator(const std::string& line) {
            const std::string t = trimmed(line);
            if (t.find('-') == std::string::npos) return false;
            return std::all_of(t.begin(), t.end(), [](char c) { return c == '|' || c == '-' || c == ':' || c == ' '; });
        }

        std::vector<std::string> splitTableRow(const std::string& line) {
            std::string t = trimmed(line);
            if (!t.empty() && t.front() == '|') t.erase(0, 1);
            if (!t.empty() && t.back() == '|' && (t.size() < 2 || t[t.size() - 2] != '\\')) t.pop_back();
            std::vector<std::string> cells;
            std::string cell;
            bool inCode = false, inMath = false;
            for (std::size_t i = 0; i < t.size(); ++i) {
                const char c = t[i];
                if (c == '\\' && i + 1 < t.size() && t[i + 1] == '|') {
                    cell += "\\|";
                    ++i;
                    continue;
                }
                if (c == '`' && !inMath) inCode = !inCode;
                if (c == '$' && !inCode) inMath = !inMath;
                if (c == '|' && !inCode && !inMath) {
                    cells.push_back(trimmed(cell));
                    cell.clear();
                    continue;
                }
                cell += c;
            }
            cells.push_back(trimmed(cell));
            return cells;
        }

        int indentOf(const std::string& line) {
            int n = 0;
            for (char c : line) {
                if (c == ' ') ++n;
                else if (c == '\t') n += 4;
                else break;
            }
            return n;
        }

        bool startsBlock(const std::string& t) {
            return t.empty() || t[0] == '#' || t[0] == '|' || startsWith(t, "$$") || startsWith(t, "```") || isBullet(t) ||
                   numberedAt(t) > 0 || isRule(t);
        }

        // --- layout ------------------------------------------------------------

        float fontPx(float designPx) {
            const ImGuiStyle& st = ImGui::GetStyle();
            return designPx * st.FontScaleMain * st.FontScaleDpi;
        }

        ImFont* faceFont(int face) {
            ImFont* f = nullptr;
            switch (face) {
                case 1: f = theme::font(theme::Weight::Bold); break;
                case 2: f = theme::font(theme::Weight::ExtraBold); break;
                case 3: f = theme::mono(); break;
                default: f = theme::font(theme::Weight::Regular); break;
            }
            return f ? f : ImGui::GetFont();
        }

        float measure(int face, float designPx, const char* begin, const char* end) {
            if (begin == end) return 0.0f;
            return faceFont(face)->CalcTextSizeA(fontPx(designPx), FLT_MAX, 0.0f, begin, end).x;
        }

        float spaceWidth(int face, float designPx) {
            static const char space[] = " ";
            return measure(face, designPx, space, space + 1);
        }

        constexpr float kLineFactor = 1.4f;

        // Places words on lines between `left` and `right`, top down.
        class Flow {
        public:
            Flow(Layout& out, float left, float right, float top, float basePx, int baseFace, ImU32 color)
                : out_(out), left_(left), right_(std::max(right, left + px(20))), basePx_(basePx), baseFace_(baseFace), color_(color),
                  x_(left), y_(top) {
                lineH_ = std::round(fontPx(basePx_) * kLineFactor);
            }

            void add(const Span& s) {
                ++span_;
                if (s.lineBreak) {
                    breakLine();
                    return;
                }
                Look look;
                look.face = s.code ? 3 : (s.bold || baseFace_ == 1 ? std::max(1, baseFace_) : baseFace_);
                look.px = s.code ? std::max(9.0f, basePx_ - 1.0f) : basePx_;
                look.color = color_;
                if (s.script != 0) {
                    look.px = std::max(8.0f, basePx_ * 0.74f);
                    look.dy = s.script > 0 ? -0.22f : 0.30f;
                }
                look.fill = s.code;
                if (!s.href.empty()) {
                    out_.links.push_back(s.href);
                    look.link = static_cast<int>(out_.links.size()) - 1;
                    look.color = theme::kAccentText;
                }
                const std::string& t = s.text;
                std::size_t i = 0;
                while (i < t.size()) {
                    const char c = t[i];
                    if (c == '\n') {
                        breakLine();
                        ++i;
                        continue;
                    }
                    if (c == ' ' || c == '\t' || c == '\r') {
                        if (!lineEmpty_) space_ = true;
                        ++i;
                        continue;
                    }
                    std::size_t j = i;
                    while (j < t.size() && t[j] != ' ' && t[j] != '\t' && t[j] != '\n' && t[j] != '\r') ++j;
                    word(t.data() + i, t.data() + j, look);
                    i = j;
                }
            }

            // Text whose spaces count (a line of a code block): broken where it
            // runs out of room, never reflowed.
            void verbatim(const std::string& line, int face, float designPx) {
                Look look;
                look.face = face;
                look.px = designPx;
                look.color = color_;
                ++span_;
                std::string t;
                for (char c : line) t += c == '\t' ? std::string(4, ' ') : std::string(1, c);
                chunks(t.data(), t.data() + t.size(), look);
                breakLine();
            }

            void breakLine() {
                widest_ = std::max(widest_, x_);
                x_ = left_;
                y_ += lineH_;
                lineEmpty_ = true;
                space_ = false;
                lastRun_ = -1;
            }

            // The y below the last line.
            float bottom() {
                if (!lineEmpty_) breakLine();
                return y_;
            }
            float widest() const { return std::max(widest_, x_); }

        private:
            struct Look {
                int face = 0;
                float px = 13;
                ImU32 color = 0;
                bool fill = false;
                int link = -1;
                float dy = 0.0f;   // in parts of the body size: scripts
            };

            void put(const char* begin, const char* end, float w, const Look& look, bool withSpace) {
                const float spaceW = withSpace ? spaceWidth(look.face, look.px) : 0.0f;
                if (lastRun_ >= 0 && lastSpan_ == span_) {
                    Run& r = out_.runs[static_cast<std::size_t>(lastRun_)];
                    if (withSpace) r.text += ' ';
                    r.text.append(begin, end);
                    r.size.x += spaceW + w;
                    x_ += spaceW + w;
                    lineEmpty_ = false;
                    return;
                }
                x_ += spaceW;
                Run r;
                const float h = fontPx(look.px);
                r.pos = ImVec2(x_, y_ + std::floor((lineH_ - fontPx(basePx_)) * 0.5f) + look.dy * fontPx(basePx_) +
                                       (look.dy == 0.0f ? (fontPx(basePx_) - h) * 0.5f : 0.0f));
                r.size = ImVec2(w, h);
                r.text.assign(begin, end);
                r.px = look.px;
                r.face = look.face;
                r.color = look.color;
                r.fill = look.fill;
                r.link = look.link;
                out_.runs.push_back(std::move(r));
                lastRun_ = static_cast<int>(out_.runs.size()) - 1;
                lastSpan_ = span_;
                x_ += w;
                lineEmpty_ = false;
            }

            void word(const char* begin, const char* end, const Look& look) {
                const float w = measure(look.face, look.px, begin, end);
                const float spaceW = space_ ? spaceWidth(look.face, look.px) : 0.0f;
                if (!lineEmpty_ && x_ + spaceW + w > right_) breakLine();
                if (w > right_ - left_) {
                    // longer than a line (a path, a URL): broken where the room ends
                    if (space_ && !lineEmpty_) x_ += spaceW;
                    space_ = false;
                    chunks(begin, end, look);
                    return;
                }
                const bool withSpace = space_ && !lineEmpty_;
                space_ = false;
                put(begin, end, w, look, withSpace);
            }

            void chunks(const char* begin, const char* end, const Look& look) {
                const char* from = begin;
                while (from < end) {
                    const float room = right_ - x_;
                    const char* to = from;
                    float w = 0.0f;
                    while (to < end) {
                        const std::string_view rest(to, static_cast<std::size_t>(end - to));
                        std::size_t n = 0;
                        nextCodepoint(rest, n);
                        if (n == 0) n = 1;
                        const float cw = measure(look.face, look.px, from, to + n);
                        if (cw > room && to > from) break;
                        if (cw > room && lineEmpty_) {   // not even one glyph fits: take it anyway
                            to += n;
                            w = cw;
                            break;
                        }
                        if (cw > room) break;
                        to += n;
                        w = cw;
                    }
                    if (to == from) {
                        breakLine();
                        continue;
                    }
                    lastRun_ = -1;
                    put(from, to, w, look, false);
                    from = to;
                    if (from < end) breakLine();
                }
            }

            Layout& out_;
            float left_, right_;
            float basePx_;
            int baseFace_;
            ImU32 color_;
            float x_, y_;
            float lineH_ = 0.0f;
            float widest_ = 0.0f;
            bool lineEmpty_ = true;
            bool space_ = false;
            int span_ = 0;
            int lastSpan_ = -1;
            int lastRun_ = -1;
        };

    } // namespace

    Document parse(const std::string& markdownIn) {
        Document doc;
        const std::vector<std::string> lines = splitLines(normalizeMathDelimiters(markdownIn));
        std::size_t i = 0;
        while (i < lines.size()) {
            const std::string t = trimmed(lines[i]);
            if (t.empty()) {
                ++i;
                continue;
            }
            if (startsWith(t, "```")) {
                // to the closing fence, or to the end while the answer is still arriving
                Block b;
                b.kind = Block::Kind::Code;
                std::string code;
                ++i;
                while (i < lines.size() && !startsWith(trimmed(lines[i]), "```")) {
                    if (!code.empty()) code += '\n';
                    code += lines[i];
                    ++i;
                }
                if (i < lines.size()) ++i;
                Span s;
                s.text = code;
                s.code = true;
                b.spans.push_back(std::move(s));
                doc.blocks.push_back(std::move(b));
                continue;
            }
            if (startsWith(t, "$$")) {
                std::string first = t.substr(2);
                std::string tex;
                const std::size_t closeSame = first.find("$$");
                ++i;
                if (closeSame != std::string::npos) {
                    tex = first.substr(0, closeSame);
                } else {
                    tex = first;
                    while (i < lines.size()) {
                        const std::string l = trimmed(lines[i]);
                        const std::size_t close = l.find("$$");
                        ++i;
                        if (close != std::string::npos) {
                            tex += "\n" + l.substr(0, close);
                            break;
                        }
                        tex += "\n" + l;
                    }
                }
                Block b;
                b.kind = Block::Kind::Math;
                appendHtml(latexToHtml(trimmed(tex), false), Style(), b.spans);
                if (!b.spans.empty()) doc.blocks.push_back(std::move(b));
                continue;
            }
            if (t[0] == '#') {
                std::size_t level = 0;
                while (level < t.size() && t[level] == '#') ++level;
                Block b;
                b.kind = Block::Kind::Heading;
                b.level = static_cast<int>(level);
                parseInline(trimmed(t.substr(level)), Style(), b.spans);
                doc.blocks.push_back(std::move(b));
                ++i;
                continue;
            }
            if (t[0] == '|') {
                bool header = true;
                while (i < lines.size() && !trimmed(lines[i]).empty() && trimmed(lines[i])[0] == '|') {
                    const std::string row = lines[i++];
                    if (isTableSeparator(row)) continue;
                    const std::vector<std::string> cells = splitTableRow(row);
                    Block b;
                    b.kind = header ? Block::Kind::TableHeader : Block::Kind::TableRow;
                    for (std::size_t c = 0; c < cells.size(); ++c) {
                        if (c) push(b.spans, " \xC2\xB7 ", Style());
                        Style st;
                        st.bold = !header && c == 0;
                        parseInline(header ? captionCase(cells[c]) : cells[c], st, b.spans);
                    }
                    doc.blocks.push_back(std::move(b));
                    header = false;
                }
                continue;
            }
            if (isRule(t)) {
                Block b;
                b.kind = Block::Kind::Rule;
                doc.blocks.push_back(std::move(b));
                ++i;
                continue;
            }
            if (isBullet(t) || numberedAt(t) > 0) {
                const std::size_t numbered = numberedAt(t);
                Block b;
                b.kind = numbered ? Block::Kind::Numbered : Block::Kind::Bullet;
                b.level = std::min(3, indentOf(lines[i]) / 2);
                std::string body = numbered ? t.substr(numbered) : t.substr(2);
                if (numbered) b.marker = t.substr(0, numbered - 1);
                ++i;
                // an item's text may go on in the lines below it
                while (i < lines.size()) {
                    const std::string next = trimmed(lines[i]);
                    if (startsBlock(next)) break;
                    body += " " + next;
                    ++i;
                }
                parseInline(body, Style(), b.spans);
                doc.blocks.push_back(std::move(b));
                continue;
            }
            std::string para;
            while (i < lines.size()) {
                const std::string pt = trimmed(lines[i]);
                if (startsBlock(pt)) break;
                if (!para.empty()) para += " ";
                para += pt;
                ++i;
            }
            Block b;
            b.kind = Block::Kind::Paragraph;
            parseInline(para, Style(), b.spans);
            doc.blocks.push_back(std::move(b));
        }
        return doc;
    }

    Document plain(const std::string& text) {
        Document doc;
        Block b;
        b.kind = Block::Kind::Paragraph;
        Span s;
        s.text = text;
        b.spans.push_back(std::move(s));
        doc.blocks.push_back(std::move(b));
        return doc;
    }

    Layout layout(const Document& doc, float width, float bodyPx, ImU32 color) {
        Layout out;
        out.width = width;
        out.scale = theme::scale();
        float y = 0.0f, widest = 0.0f;
        bool first = true;
        Block::Kind previous = Block::Kind::Paragraph;
        for (const Block& b : doc.blocks) {
            const bool list = b.kind == Block::Kind::Bullet || b.kind == Block::Kind::Numbered;
            const bool row = b.kind == Block::Kind::TableHeader || b.kind == Block::Kind::TableRow;
            if (!first) {
                const bool listBefore = previous == Block::Kind::Bullet || previous == Block::Kind::Numbered;
                const bool rowBefore = previous == Block::Kind::TableHeader || previous == Block::Kind::TableRow;
                if (b.kind == Block::Kind::Heading) y += px(14);
                else if (previous == Block::Kind::Heading) y += px(4);
                else if (list && listBefore) y += px(3);
                else if (row && rowBefore) y += px(3);
                else y += px(8);
            }
            first = false;
            previous = b.kind;
            switch (b.kind) {
                case Block::Kind::Heading: {
                    const float size = b.level <= 1 ? 20.0f : (b.level == 2 ? 15.0f : 13.0f);
                    Flow flow(out, 0.0f, width, y, size, 2, color);
                    for (const Span& s : b.spans) flow.add(s);
                    y = flow.bottom();
                    widest = std::max(widest, flow.widest());
                    break;
                }
                case Block::Kind::Bullet:
                case Block::Kind::Numbered: {
                    const float indent = px(14.0f * static_cast<float>(b.level));
                    const std::string marker = b.kind == Block::Kind::Bullet ? std::string("\xE2\x80\xA2") : b.marker;
                    const float markerW = measure(0, bodyPx, marker.data(), marker.data() + marker.size());
                    const float column = std::max(px(14), markerW + px(5));
                    const float lineH = std::round(fontPx(bodyPx) * kLineFactor);
                    Run m;
                    m.pos = ImVec2(indent + (b.kind == Block::Kind::Bullet ? px(2) : 0.0f), y + std::floor((lineH - fontPx(bodyPx)) * 0.5f));
                    m.size = ImVec2(markerW, fontPx(bodyPx));
                    m.text = marker;
                    m.px = bodyPx;
                    m.color = color;
                    out.runs.push_back(std::move(m));
                    Flow flow(out, indent + column, width, y, bodyPx, 0, color);
                    for (const Span& s : b.spans) flow.add(s);
                    y = std::max(flow.bottom(), y + lineH);
                    widest = std::max(widest, flow.widest());
                    break;
                }
                case Block::Kind::Code: {
                    const float pad = px(8);
                    const float top = y;
                    Flow flow(out, pad, width - pad, y + pad, 11.5f, 3, theme::kNeutral900);
                    for (const Span& s : b.spans)
                        for (const std::string& line : split(s.text, '\n')) flow.verbatim(line, 3, 11.5f);
                    y = flow.bottom() + pad;
                    Fill f;
                    f.min = ImVec2(0.0f, top);
                    f.max = ImVec2(width, y);
                    f.color = theme::kSurface;
                    out.fills.push_back(f);
                    widest = width;
                    break;
                }
                case Block::Kind::Math: {
                    const float pad = px(10);
                    const float top = y;
                    Flow flow(out, pad, width - pad, y + pad, 15.0f, 0, color);
                    for (const Span& s : b.spans) flow.add(s);
                    y = flow.bottom() + pad;
                    Fill f;
                    f.min = ImVec2(0.0f, top);
                    f.max = ImVec2(width, y);
                    f.color = theme::kSurface;
                    f.outline = true;
                    out.fills.push_back(f);
                    widest = width;
                    break;
                }
                case Block::Kind::TableHeader:
                case Block::Kind::TableRow: {
                    const bool header = b.kind == Block::Kind::TableHeader;
                    Flow flow(out, 0.0f, width, y, header ? 10.0f : bodyPx, 0, header ? theme::kNeutral600 : color);
                    for (const Span& s : b.spans) flow.add(s);
                    y = flow.bottom() + px(3);
                    Fill f;
                    f.min = ImVec2(0.0f, y);
                    f.max = ImVec2(width, y + theme::crispPen(header ? 2.0f : 1.0f));
                    f.color = theme::kDivider;
                    out.fills.push_back(f);
                    y = f.max.y;
                    widest = width;
                    break;
                }
                case Block::Kind::Rule: {
                    Fill f;
                    f.min = ImVec2(0.0f, y);
                    f.max = ImVec2(width, y + theme::crispPen(1.0f));
                    f.color = theme::kDivider;
                    out.fills.push_back(f);
                    y = f.max.y;
                    widest = width;
                    break;
                }
                case Block::Kind::Paragraph: {
                    Flow flow(out, 0.0f, width, y, bodyPx, 0, color);
                    for (const Span& s : b.spans) flow.add(s);
                    y = flow.bottom();
                    widest = std::max(widest, flow.widest());
                    break;
                }
            }
        }
        out.size = ImVec2(std::min(width, std::ceil(widest)), std::ceil(y));
        return out;
    }

    int draw(ImDrawList* dl, ImVec2 origin, const Layout& layout) {
        const ImVec2 clipMin = dl->GetClipRectMin(), clipMax = dl->GetClipRectMax();
        if (origin.y > clipMax.y || origin.y + layout.size.y < clipMin.y) return -1;
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        const bool hoverable = ImGui::IsWindowHovered();
        int hovered = -1;
        for (const Fill& f : layout.fills) {
            const ImVec2 a(theme::snap(origin.x + f.min.x), theme::snap(origin.y + f.min.y));
            const ImVec2 b(theme::snap(origin.x + f.max.x), theme::snap(origin.y + f.max.y));
            if (b.y < clipMin.y || a.y > clipMax.y) continue;
            dl->AddRectFilled(a, b, f.color);
            if (f.outline) dl->AddRect(a, b, theme::kDivider, 0.0f, ImDrawFlags_None, theme::crispPen(1.0f));
        }
        if (hoverable)
            for (const Run& r : layout.runs) {
                if (r.link < 0) continue;
                const ImVec2 a(origin.x + r.pos.x, origin.y + r.pos.y);
                if (mouse.x >= a.x && mouse.x < a.x + r.size.x && mouse.y >= a.y - px(2) && mouse.y < a.y + r.size.y + px(2))
                    hovered = r.link;
            }
        for (const Run& r : layout.runs) {
            const ImVec2 a(theme::snap(origin.x + r.pos.x), theme::snap(origin.y + r.pos.y));
            if (a.y + r.size.y < clipMin.y || a.y > clipMax.y) continue;
            if (r.fill)
                dl->AddRectFilled(ImVec2(a.x - px(2), a.y - px(1)), ImVec2(a.x + r.size.x + px(2), a.y + r.size.y + px(1)), theme::kSurface);
            const bool hot = r.link >= 0 && r.link == hovered;
            dl->AddText(faceFont(r.face), fontPx(r.px), a, hot ? theme::kAccent700 : r.color, r.text.data(), r.text.data() + r.text.size());
            if (hot) {
                const float t = theme::crispPen(1.0f);
                dl->AddRectFilled(ImVec2(a.x, a.y + r.size.y), ImVec2(a.x + r.size.x, a.y + r.size.y + t), theme::kAccent700);
            }
        }
        return hovered;
    }

} // namespace sirius::app::gui::assistant_markdown
