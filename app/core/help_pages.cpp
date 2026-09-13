#include "core/help_pages.hpp"

#include "core/app_paths.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <utility>

namespace sirius::app {

    namespace fs = std::filesystem;

    // --- small string helpers -------------------------------------------------

    namespace {
        // Design tokens (docs/design/README.md); the core is Qt-free, so the
        // HTML carries them as literals.
        constexpr const char* kSurface = "#eae9e9";
        constexpr const char* kText = "#201e1d";
        constexpr const char* kDivider = "#a6a5a4";
        constexpr const char* kAccent = "#ec3013";
        constexpr const char* kNeutral600 = "#7d7979";
        constexpr const char* kNeutral800 = "#444141";

        std::string trim(const std::string& s) {
            const auto a = s.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) return {};
            const auto b = s.find_last_not_of(" \t\r\n");
            return s.substr(a, b - a + 1);
        }

        bool startsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        std::string escapeHtml(const std::string& s) {
            std::string out;
            out.reserve(s.size());
            for (char c : s) {
                switch (c) {
                    case '&': out += "&amp;"; break;
                    case '<': out += "&lt;"; break;
                    case '>': out += "&gt;"; break;
                    case '"': out += "&quot;"; break;
                    default: out += c;
                }
            }
            return out;
        }

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
            bool inMath = false;
            for (std::size_t i = 0; i < line.size(); ++i) {
                const char c = line[i];
                if (c == '\\' && i + 1 < line.size()) {
                    cur += c;
                    cur += line[++i];
                    continue;
                }
                if (c == '$') inMath = !inMath;
                if (c == '|' && !inMath) {
                    cells.push_back(cur);
                    cur.clear();
                } else {
                    cur += c;
                }
            }
            cells.push_back(cur);
            // leading / trailing pipes produce empty edge cells
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

        // Position of the last "$...$" (single dollars) in `s`, or npos.
        std::pair<std::size_t, std::size_t> lastInlineMath(const std::string& s) {
            std::size_t end = s.size();
            while (end > 0) {
                const std::size_t close = s.rfind('$', end - 1);
                if (close == std::string::npos || close == 0) break;
                if (close > 0 && s[close - 1] == '\\') {
                    end = close;
                    continue;
                }
                const std::size_t open = s.rfind('$', close - 1);
                if (open == std::string::npos) break;
                if (open > 0 && s[open - 1] == '\\') {
                    end = open;
                    continue;
                }
                if (open + 1 == close) {
                    end = open;
                    continue;
                }   // "$$"
                return {open, close};
            }
            return {std::string::npos, std::string::npos};
        }
    } // namespace

    // --- LaTeX -> HTML ----------------------------------------------------------

    namespace {

        struct Node {
            enum class Kind { Text,
                              Seq,
                              Frac,
                              Sup,
                              Sub,
                              Cases };
            Kind kind = Kind::Text;
            std::string html;                       // Text
            std::vector<std::unique_ptr<Node>> kids;   // Seq; Frac (num, den); Sup/Sub (one); Cases (rows: Seq of cells)
            bool operand = true;                    // Text: a letter/number/closing bracket (for spacing of binary ops)
        };
        using NodePtr = std::unique_ptr<Node>;

        NodePtr text(std::string html, bool operand = true) {
            auto n = std::make_unique<Node>();
            n->kind = Node::Kind::Text;
            n->html = std::move(html);
            n->operand = operand;
            return n;
        }

        NodePtr seq() {
            auto n = std::make_unique<Node>();
            n->kind = Node::Kind::Seq;
            return n;
        }

        const std::map<std::string, std::string>& symbolTable() {
            static const std::map<std::string, std::string> t = {
                {"cdot", "·"},
                {"times", "×"},
                {"in", "∈"},
                {"notin", "∉"},
                {"mid", "∣"},
                {"ast", "∗"},
                {"star", "⋆"},
                {"nabla", "∇"},
                {"rightarrow", "→"},
                {"to", "→"},
                {"leftarrow", "←"},
                {"Rightarrow", "⇒"},
                {"leftrightarrow", "↔"},
                {"mapsto", "↦"},
                {"sum", "∑"},
                {"prod", "∏"},
                {"int", "∫"},
                {"infty", "∞"},
                {"pm", "±"},
                {"mp", "∓"},
                {"leq", "≤"},
                {"le", "≤"},
                {"geq", "≥"},
                {"ge", "≥"},
                {"neq", "≠"},
                {"ne", "≠"},
                {"approx", "≈"},
                {"sim", "∼"},
                {"simeq", "≃"},
                {"equiv", "≡"},
                {"propto", "∝"},
                {"partial", "∂"},
                {"ldots", "…"},
                {"cdots", "⋯"},
                {"dots", "…"},
                {"vdots", "⋮"},
                {"langle", "⟨"},
                {"rangle", "⟩"},
                {"lfloor", "⌊"},
                {"rfloor", "⌋"},
                {"lceil", "⌈"},
                {"rceil", "⌉"},
                {"circ", "∘"},
                {"forall", "∀"},
                {"exists", "∃"},
                {"subset", "⊂"},
                {"subseteq", "⊆"},
                {"cup", "∪"},
                {"cap", "∩"},
                {"emptyset", "∅"},
                {"prime", "′"},
                {"deg", "°"},
                {"ell", "ℓ"},
                {"hbar", "ℏ"},
                {"Re", "ℜ"},
                {"Im", "ℑ"},
                {"otimes", "⊗"},
                {"oplus", "⊕"},
                {"perp", "⊥"},
                {"parallel", "∥"},
                {"angle", "∠"},
                {"star", "⋆"},
                {"lvert", "|"},
                {"rvert", "|"},
                {"lVert", "‖"},
                {"rVert", "‖"},
                {"vert", "|"},
                {"Vert", "‖"},
                // Greek
                {"alpha", "α"},
                {"beta", "β"},
                {"gamma", "γ"},
                {"delta", "δ"},
                {"epsilon", "ε"},
                {"varepsilon", "ε"},
                {"zeta", "ζ"},
                {"eta", "η"},
                {"theta", "θ"},
                {"vartheta", "ϑ"},
                {"iota", "ι"},
                {"kappa", "κ"},
                {"lambda", "λ"},
                {"mu", "μ"},
                {"nu", "ν"},
                {"xi", "ξ"},
                {"pi", "π"},
                {"rho", "ρ"},
                {"varrho", "ϱ"},
                {"sigma", "σ"},
                {"tau", "τ"},
                {"upsilon", "υ"},
                {"phi", "φ"},
                {"varphi", "ϕ"},
                {"chi", "χ"},
                {"psi", "ψ"},
                {"omega", "ω"},
                {"Gamma", "Γ"},
                {"Delta", "Δ"},
                {"Theta", "Θ"},
                {"Lambda", "Λ"},
                {"Xi", "Ξ"},
                {"Pi", "Π"},
                {"Sigma", "Σ"},
                {"Upsilon", "Υ"},
                {"Phi", "Φ"},
                {"Psi", "Ψ"},
                {"Omega", "Ω"},
            };
            return t;
        }

        // Commands that read as upright words.
        const std::map<std::string, std::string>& operatorTable() {
            static const std::map<std::string, std::string> t = {
                {"max", "max"},
                {"min", "min"},
                {"arg", "arg"},
                {"argmax", "arg max"},
                {"argmin", "arg min"},
                {"log", "log"},
                {"ln", "ln"},
                {"exp", "exp"},
                {"sin", "sin"},
                {"cos", "cos"},
                {"tan", "tan"},
                {"lim", "lim"},
                {"det", "det"},
                {"sup", "sup"},
                {"inf", "inf"},
                {"clip", "clip"},
            };
            return t;
        }

        const std::map<std::string, std::string>& accentTable() {
            static const std::map<std::string, std::string> t = {
                {"tilde", "\xCC\x83"},
                {"hat", "\xCC\x82"},
                {"bar", "\xCC\x84"},
                {"vec", "\xE2\x83\x97"},
                {"dot", "\xCC\x87"},
                {"ddot", "\xCC\x88"},
                {"overline", "\xCC\x85"},
            };
            return t;
        }

        // Whether a rendered single symbol is one glyph (a base for a combining accent).
        bool singleGlyph(const std::string& s) {
            // strip tags
            std::string plain;
            bool tag = false;
            for (char c : s) {
                if (c == '<') tag = true;
                else if (c == '>') tag = false;
                else if (!tag) plain += c;
            }
            std::size_t n = 0;
            for (unsigned char c : plain)
                if ((c & 0xC0) != 0x80) ++n;
            return n == 1;
        }

        class Parser {
        public:
            explicit Parser(const std::string& s) : s_(s) {}

            NodePtr parseAll() { return parseSequence(0); }

        private:
            // Parses until the closing brace at depth or end of input.
            NodePtr parseSequence(int depth, const char* stopCommand = nullptr) {
                NodePtr out = seq();
                while (pos_ < s_.size()) {
                    const char c = s_[pos_];
                    if (c == '}') {
                        if (depth > 0) {
                            ++pos_;
                            return out;
                        }
                        ++pos_;   // stray
                        continue;
                    }
                    if (c == '&' && stopCommand) {   // cases column separator, handled by caller
                        return out;
                    }
                    if (stopCommand && (startsWith(s_.substr(pos_), stopCommand) || s_.compare(pos_, 2, "\\\\") == 0))
                        return out;
                    NodePtr item = parseItem(depth, stopCommand);
                    if (item) attach(*out, std::move(item));
                }
                return out;
            }

            // Sup/Sub attach to the previous item.
            void attach(Node& seqNode, NodePtr item) {
                if (item->kind == Node::Kind::Sup || item->kind == Node::Kind::Sub) {
                    if (seqNode.kids.empty()) seqNode.kids.push_back(text(""));
                }
                seqNode.kids.push_back(std::move(item));
            }

            NodePtr parseArgument(int depth) {
                skipSpaces();
                if (pos_ >= s_.size()) return text("");
                if (s_[pos_] == '{') {
                    ++pos_;
                    return parseSequence(depth + 1);
                }
                NodePtr item = parseItem(depth, nullptr);
                return item ? std::move(item) : text("");
            }

            void skipSpaces() {
                while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) ++pos_;
            }

            std::string readCommandName() {
                std::string name;
                while (pos_ < s_.size() && std::isalpha(static_cast<unsigned char>(s_[pos_]))) name += s_[pos_++];
                return name;
            }

            std::string readRawGroup() {   // after '{' consumed? no: expects '{'
                skipSpaces();
                std::string out;
                if (pos_ >= s_.size() || s_[pos_] != '{') {
                    if (pos_ < s_.size()) out += s_[pos_++];
                    return out;
                }
                ++pos_;
                int depth = 1;
                while (pos_ < s_.size()) {
                    const char c = s_[pos_++];
                    if (c == '{') ++depth;
                    else if (c == '}') {
                        if (--depth == 0) break;
                    }
                    out += c;
                }
                return out;
            }

            NodePtr parseItem(int depth, const char* stopCommand) {
                const char c = s_[pos_];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    ++pos_;
                    return nullptr;
                }
                if (c == '{') {
                    ++pos_;
                    return parseSequence(depth + 1);
                }
                if (c == '^' || c == '_') {
                    ++pos_;
                    NodePtr arg = parseArgument(depth);
                    auto n = std::make_unique<Node>();
                    n->kind = c == '^' ? Node::Kind::Sup : Node::Kind::Sub;
                    n->kids.push_back(std::move(arg));
                    return n;
                }
                if (c == '\\') return parseCommand(depth, stopCommand);
                ++pos_;
                return parseChar(c);
            }

            NodePtr parseChar(char c) {
                switch (c) {
                    case '=': return text(" = ", false);
                    case '+': return text(" + ", false);
                    case '<': return text(" &lt; ", false);
                    case '>': return text(" &gt; ", false);
                    case '-': return text(lastWasOperand() ? " − " : "−", false);
                    case '*': return text("∗", false);
                    case '\'': return text("′");
                    case ',': return text(", ", false);
                    case '(':
                    case '[': return text(std::string(1, c), false);
                    case ')':
                    case ']':
                    case '|': return text(std::string(1, c), true);
                    case '&': return text("&amp;", false);
                    default: break;
                }
                if (std::isalpha(static_cast<unsigned char>(c))) return text(std::string("<i>") + c + "</i>");
                return text(escapeHtml(std::string(1, c)), std::isdigit(static_cast<unsigned char>(c)) != 0);
            }

            bool lastWasOperand() const { return lastOperand_; }

            NodePtr parseCommand(int depth, const char* stopCommand) {
                ++pos_;   // backslash
                if (pos_ >= s_.size()) return text("\\");
                const char first = s_[pos_];
                if (!std::isalpha(static_cast<unsigned char>(first))) {
                    ++pos_;
                    switch (first) {
                        case ',': return text("\xE2\x80\x89", false);          // thin space
                        case ';':
                        case ':': return text("\xE2\x80\x85", false); // four-per-em
                        case '!': return nullptr;
                        case '|': return text("‖");
                        case '{': return text("{", false);
                        case '}': return text("}", true);
                        case '\\': return text("<br>", false);
                        case ' ': return text(" ", false);
                        case '&': return text("&amp;", false);
                        case '%': return text("%");
                        case '$': return text("$");
                        case '#': return text("#");
                        default: return text(escapeHtml(std::string(1, first)));
                    }
                }
                const std::string name = readCommandName();
                if (stopCommand && name == "end") {   // consumed by the caller
                    pos_ -= name.size() + 1;
                    return nullptr;
                }
                if (name == "frac" || name == "dfrac" || name == "tfrac") {
                    auto n = std::make_unique<Node>();
                    n->kind = Node::Kind::Frac;
                    n->kids.push_back(parseArgument(depth));
                    n->kids.push_back(parseArgument(depth));
                    return n;
                }
                if (name == "sqrt") {
                    NodePtr arg = parseArgument(depth);
                    NodePtr out = seq();
                    out->kids.push_back(text("√", false));
                    out->kids.push_back(text("<span style=\"text-decoration: overline\">", false));
                    out->kids.push_back(std::move(arg));
                    out->kids.push_back(text("</span>"));
                    return out;
                }
                if (name == "text" || name == "mathrm" || name == "operatorname" || name == "textrm" || name == "mbox") {
                    const std::string raw = readRawGroup();
                    return text("<span style=\"font-style: normal\">" + escapeHtml(raw) + "</span>");
                }
                if (name == "mathbf" || name == "boldsymbol" || name == "bm") {
                    NodePtr arg = parseArgument(depth);
                    // upright bold: drop the italics of the letters inside
                    std::string inner = renderInline(*arg);
                    for (const char* tag : {"<i>", "</i>"})
                        for (std::size_t p = inner.find(tag); p != std::string::npos; p = inner.find(tag, p))
                            inner.erase(p, std::char_traits<char>::length(tag));
                    return text("<b style=\"font-style: normal\">" + inner + "</b>");
                }
                if (name == "mathbb") {
                    const std::string raw = trim(readRawGroup());
                    static const std::map<std::string, std::string> bb = {{"N", "ℕ"}, {"R", "ℝ"}, {"Z", "ℤ"}, {"C", "ℂ"}, {"Q", "ℚ"}, {"E", "𝔼"}, {"P", "ℙ"}};
                    auto it = bb.find(raw);
                    return text(it != bb.end() ? it->second : "<b>" + escapeHtml(raw) + "</b>");
                }
                if (name == "mathcal" || name == "mathit" || name == "mathsf") {
                    NodePtr arg = parseArgument(depth);
                    return arg;
                }
                if (auto acc = accentTable().find(name); acc != accentTable().end()) {
                    NodePtr arg = parseArgument(depth);
                    return applyAccent(std::move(arg), acc->second, name);
                }
                if (name == "left" || name == "right" || name == "big" || name == "Big" || name == "bigl" ||
                    name == "bigr" || name == "Bigl" || name == "Bigr" || name == "bigg" || name == "Bigg") {
                    skipSpaces();
                    if (pos_ < s_.size()) {
                        if (s_[pos_] == '.') {
                            ++pos_;
                            return nullptr;
                        }
                        if (s_[pos_] == '\\') {
                            return parseCommand(depth, stopCommand);   // \left\| etc.
                        }
                        const char d = s_[pos_++];
                        return text(escapeHtml(std::string(1, d)), d == ')' || d == ']' || d == '|' || d == '}');
                    }
                    return nullptr;
                }
                if (name == "quad") return text("\xE2\x80\x83", false);
                if (name == "qquad") return text("\xE2\x80\x83\xE2\x80\x83", false);
                if (name == "displaystyle" || name == "textstyle" || name == "limits" || name == "nolimits") return nullptr;
                if (name == "begin") {
                    const std::string env = trim(readRawGroup());
                    if (env == "cases" || env == "aligned" || env == "array" || env == "matrix" || env == "pmatrix") {
                        if (env == "array") readRawGroup();   // column spec
                        return parseCases(depth, env);
                    }
                    return nullptr;
                }
                if (name == "end") {
                    readRawGroup();
                    return nullptr;
                }
                if (auto op = operatorTable().find(name); op != operatorTable().end())
                    return text("<span style=\"font-style: normal\">" + op->second + "</span>");
                if (auto sym = symbolTable().find(name); sym != symbolTable().end()) {
                    const bool binary = name == "cdot" || name == "times" || name == "in" || name == "mid" ||
                                        name == "rightarrow" || name == "to" || name == "leftarrow" ||
                                        name == "Rightarrow" || name == "leq" || name == "le" || name == "geq" ||
                                        name == "ge" || name == "neq" || name == "ne" || name == "approx" ||
                                        name == "propto" || name == "equiv" || name == "sim" || name == "mapsto";
                    if (binary) return text(" " + sym->second + " ", false);
                    const bool operand = name != "sum" && name != "prod" && name != "int" && name != "nabla" &&
                                         name != "partial" && name != "pm" && name != "mp";
                    return text(sym->second, operand);
                }
                // unknown: keep the name upright so the author sees it
                return text("<span style=\"font-style: normal\">" + escapeHtml(name) + "</span>");
            }

            NodePtr applyAccent(NodePtr arg, const std::string& combining, const std::string& name) {
                std::string rendered = renderInline(*arg);
                if (singleGlyph(rendered)) {
                    // insert the combining mark before the closing tag(s)
                    const std::size_t close = rendered.find("</");
                    if (close == std::string::npos) rendered += combining;
                    else rendered.insert(close, combining);
                    return text(rendered);
                }
                // multi-glyph: prefix with the accent as a separate symbol
                static const std::map<std::string, std::string> spaced = {
                    {"tilde", "˜"}, {"hat", "ˆ"}, {"bar", "¯"}, {"vec", "→"}, {"dot", "˙"}, {"ddot", "¨"}, {"overline", "¯"}};
                auto it = spaced.find(name);
                NodePtr out = seq();
                out->kids.push_back(text(it != spaced.end() ? it->second : "", false));
                out->kids.push_back(std::move(arg));
                return out;
            }

            NodePtr parseCases(int depth, const std::string& env) {
                auto n = std::make_unique<Node>();
                n->kind = Node::Kind::Cases;
                n->html = env;
                NodePtr row = seq();
                while (pos_ < s_.size()) {
                    NodePtr cell = parseSequence(depth, "\\end");
                    row->kids.push_back(std::move(cell));
                    if (pos_ >= s_.size()) break;
                    if (s_[pos_] == '&') {
                        ++pos_;
                        continue;
                    }
                    // "\\" row break or "\end"
                    if (startsWith(s_.substr(pos_), "\\end")) {
                        pos_ += 4;
                        readRawGroup();
                        break;
                    }
                    if (startsWith(s_.substr(pos_), "\\\\")) {
                        pos_ += 2;
                        n->kids.push_back(std::move(row));
                        row = seq();
                        continue;
                    }
                    ++pos_;   // safety
                }
                if (!row->kids.empty()) n->kids.push_back(std::move(row));
                return n;
            }

        public:
            // Row-break "\\" inside cases is parsed as a Text("<br>") by parseCommand
            // when not in a cases context; inside parseCases the stop check runs first.

            static std::string renderInline(const Node& n) {
                switch (n.kind) {
                    case Node::Kind::Text: return n.html;
                    case Node::Kind::Seq: {
                        std::string out;
                        for (const auto& k : n.kids) out += renderInline(*k);
                        return out;
                    }
                    case Node::Kind::Frac: {
                        const std::string num = renderInline(*n.kids[0]);
                        const std::string den = renderInline(*n.kids[1]);
                        return "<sup>" + num + "</sup>\xE2\x81\x84<sub>" + den + "</sub>";
                    }
                    case Node::Kind::Sup: return "<sup>" + renderInline(*n.kids[0]) + "</sup>";
                    case Node::Kind::Sub: return "<sub>" + renderInline(*n.kids[0]) + "</sub>";
                    case Node::Kind::Cases: {
                        std::string out = n.html == "cases" ? "{ " : "";
                        for (std::size_t r = 0; r < n.kids.size(); ++r) {
                            if (r) out += "; ";
                            const Node& row = *n.kids[r];
                            for (std::size_t c = 0; c < row.kids.size(); ++c) {
                                if (c) out += (n.html == "cases" ? " <span style=\"font-style: normal\">if</span> " : " ");
                                out += renderInline(*row.kids[c]);
                            }
                        }
                        return out;
                    }
                }
                return {};
            }

            // Display mode: a one-row table whose cells alternate inline runs
            // and stacked fractions / cases so the formula keeps one baseline.
            static std::string renderDisplay(const Node& root) {
                std::vector<const Node*> flat;
                flatten(root, flat);
                const bool stacked = std::any_of(flat.begin(), flat.end(), [](const Node* n) {
                    return n->kind == Node::Kind::Frac || n->kind == Node::Kind::Cases;
                });
                if (!stacked) return "<span style=\"font-size: 15px\">" + renderInline(root) + "</span>";
                std::string out = "<table cellspacing=\"0\" cellpadding=\"2\" border=\"0\" style=\"font-size: 15px\"><tr>";
                std::string run;
                auto flush = [&]() {
                    if (!run.empty()) out += "<td valign=\"middle\" style=\"white-space: nowrap\">" + run + "</td>";
                    run.clear();
                };
                for (const Node* n : flat) {
                    if (n->kind == Node::Kind::Frac) {
                        flush();
                        out += "<td valign=\"middle\">" + fractionTable(*n) + "</td>";
                    } else if (n->kind == Node::Kind::Cases) {
                        flush();
                        out += "<td valign=\"middle\">" + casesTable(*n) + "</td>";
                    } else {
                        run += renderInline(*n);
                    }
                }
                flush();
                out += "</tr></table>";
                return out;
            }

        private:
            static void flatten(const Node& n, std::vector<const Node*>& out) {
                if (n.kind == Node::Kind::Seq) {
                    for (const auto& k : n.kids) flatten(*k, out);
                } else {
                    out.push_back(&n);
                }
            }

            static std::string fractionTable(const Node& frac) {
                const std::string num = renderInline(*frac.kids[0]);
                const std::string den = renderInline(*frac.kids[1]);
                return std::string("<table cellspacing=\"0\" cellpadding=\"1\" border=\"0\"><tr><td align=\"center\" style=\"border-bottom: 1px solid ") +
                       kText + "; white-space: nowrap\">" + num + "</td></tr><tr><td align=\"center\" style=\"white-space: nowrap\">" +
                       den + "</td></tr></table>";
            }

            static std::string casesTable(const Node& cases) {
                std::string out = "<table cellspacing=\"0\" cellpadding=\"2\" border=\"0\"><tr>";
                const std::size_t rows = cases.kids.size();
                if (cases.html == "cases")
                    out += "<td rowspan=\"" + std::to_string(std::max<std::size_t>(rows, 1)) +
                           "\" valign=\"middle\" style=\"font-size: " + std::to_string(15 + 8 * rows) + "px; font-weight: 200\">{</td>";
                for (std::size_t r = 0; r < rows; ++r) {
                    if (r) out += "<tr>";
                    const Node& row = *cases.kids[r];
                    for (std::size_t c = 0; c < row.kids.size(); ++c)
                        out += "<td valign=\"middle\" style=\"white-space: nowrap; padding-left: 8px\">" + renderInline(*row.kids[c]) + "</td>";
                    out += "</tr>";
                }
                out += "</table>";
                return out;
            }

            const std::string& s_;
            std::size_t pos_ = 0;
            bool lastOperand_ = false;
        };

    } // namespace

    std::string latexToHtml(const std::string& tex, bool display) {
        const std::string t = trim(tex);
        if (t.empty()) return {};
        Parser parser(t);
        NodePtr root = parser.parseAll();
        // spacing of unary minus: recompute operand flags in sequence order
        // (the parser tracks it through lastOperand_ only for direct text; a
        // second pass keeps " − " between operands and "−" after operators).
        std::function<void(Node&, bool&)> fix = [&](Node& n, bool& lastOperand) {
            switch (n.kind) {
                case Node::Kind::Text:
                    if (n.html == " − " || n.html == "−") n.html = lastOperand ? " − " : "−";
                    if (!n.html.empty()) lastOperand = n.operand;
                    break;
                case Node::Kind::Seq:
                    for (auto& k : n.kids) fix(*k, lastOperand);
                    break;
                case Node::Kind::Frac: {
                    bool a = false, b = false;
                    fix(*n.kids[0], a);
                    fix(*n.kids[1], b);
                    lastOperand = true;
                    break;
                }
                case Node::Kind::Sup:
                case Node::Kind::Sub: {
                    bool a = false;
                    fix(*n.kids[0], a);
                    lastOperand = true;
                    break;
                }
                case Node::Kind::Cases:
                    for (auto& row : n.kids)
                        for (auto& cell : row->kids) {
                            bool a = false;
                            fix(*cell, a);
                        }
                    lastOperand = true;
                    break;
            }
        };
        bool lastOperand = false;
        fix(*root, lastOperand);
        return display ? Parser::renderDisplay(*root) : Parser::renderInline(*root);
    }

    // --- Markdown -> HTML --------------------------------------------------------

    namespace {

        std::string inlineMarkdownToHtml(const std::string& text, const std::string& baseDir);

        std::string linkOrImage(const std::string& s, std::size_t& i, const std::string& baseDir, bool image) {
            // s[i] == '[' (image: s[i-1] == '!')
            const std::size_t close = s.find(']', i);
            if (close == std::string::npos || close + 1 >= s.size() || s[close + 1] != '(') return {};
            const std::size_t end = s.find(')', close + 2);
            if (end == std::string::npos) return {};
            const std::string label = s.substr(i + 1, close - i - 1);
            std::string target = s.substr(close + 2, end - close - 2);
            i = end;
            if (image) {
                if (!baseDir.empty() && target.find("://") == std::string::npos && !fs::path(target).is_absolute())
                    target = (fs::path(baseDir) / target).generic_string();   // '/' separators: a URL, not a native path
                return "<img src=\"" + escapeHtml(target) + "\" alt=\"" + escapeHtml(label) + "\" style=\"max-width: 100%\">";
            }
            return "<a href=\"" + escapeHtml(target) + "\" style=\"color: " + kAccent + "\">" +
                   inlineMarkdownToHtml(label, baseDir) + "</a>";
        }

        std::string inlineMarkdownToHtml(const std::string& s, const std::string& baseDir) {
            std::string out;
            for (std::size_t i = 0; i < s.size(); ++i) {
                const char c = s[i];
                if (c == '\\' && i + 1 < s.size() && (s[i + 1] == '$' || s[i + 1] == '*' || s[i + 1] == '`' || s[i + 1] == '|')) {
                    out += s[++i];
                    continue;
                }
                if (c == '$') {
                    const bool dbl = i + 1 < s.size() && s[i + 1] == '$';
                    const std::string fence = dbl ? "$$" : "$";
                    const std::size_t end = s.find(fence, i + fence.size());
                    if (end != std::string::npos) {
                        out += latexToHtml(s.substr(i + fence.size(), end - i - fence.size()), false);
                        i = end + fence.size() - 1;
                        continue;
                    }
                }
                if (c == '`') {
                    const std::size_t end = s.find('`', i + 1);
                    if (end != std::string::npos) {
                        out += "<code style=\"font-family: monospace; background: " + std::string(kSurface) + "\">" +
                               escapeHtml(s.substr(i + 1, end - i - 1)) + "</code>";
                        i = end;
                        continue;
                    }
                }
                if (c == '*' && i + 1 < s.size() && s[i + 1] == '*') {
                    const std::size_t end = s.find("**", i + 2);
                    if (end != std::string::npos) {
                        out += "<b>" + inlineMarkdownToHtml(s.substr(i + 2, end - i - 2), baseDir) + "</b>";
                        i = end + 1;
                        continue;
                    }
                }
                if (c == '*') {
                    const std::size_t end = s.find('*', i + 1);
                    if (end != std::string::npos && end > i + 1) {
                        out += "<i>" + inlineMarkdownToHtml(s.substr(i + 1, end - i - 1), baseDir) + "</i>";
                        i = end;
                        continue;
                    }
                }
                if (c == '!' && i + 1 < s.size() && s[i + 1] == '[') {
                    std::size_t j = i + 1;
                    const std::string html = linkOrImage(s, j, baseDir, true);
                    if (!html.empty()) {
                        out += html;
                        i = j;
                        continue;
                    }
                }
                if (c == '[') {
                    std::size_t j = i;
                    const std::string html = linkOrImage(s, j, baseDir, false);
                    if (!html.empty()) {
                        out += html;
                        i = j;
                        continue;
                    }
                }
                if (c == '<') {
                    // pass through a few harmless tags, escape the rest
                    static const char* allowed[] = {"<br>", "<br/>", "<br />", "<sub>", "</sub>", "<sup>", "</sup>", "<b>", "</b>", "<i>", "</i>"};
                    bool passed = false;
                    for (const char* tag : allowed) {
                        const std::size_t n = std::char_traits<char>::length(tag);
                        if (s.compare(i, n, tag) == 0) {
                            out += tag;
                            i += n - 1;
                            passed = true;
                            break;
                        }
                    }
                    if (passed) continue;
                    out += "&lt;";
                    continue;
                }
                if (c == '&') {
                    out += "&amp;";
                    continue;
                }
                if (c == '>') {
                    out += "&gt;";
                    continue;
                }
                out += c;
            }
            return out;
        }

        std::string tableToHtml(const std::vector<std::string>& rows, const std::string& baseDir) {
            std::string html = std::string("<table cellspacing=\"0\" cellpadding=\"6\" width=\"100%\" style=\"border-collapse: collapse\">");
            bool header = true;
            for (const std::string& line : rows) {
                if (isTableSeparator(line)) continue;
                const std::vector<std::string> cells = splitTableRow(line);
                html += "<tr>";
                for (std::size_t i = 0; i < cells.size(); ++i) {
                    if (header)
                        html += "<td style=\"border-bottom: 2px solid " + std::string(kDivider) + "; color: " + kNeutral600 +
                                "; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em\">" +
                                inlineMarkdownToHtml(cells[i], baseDir) + "</td>";
                    else
                        html += std::string("<td valign=\"top\" style=\"border-bottom: 1px solid ") + kDivider +
                                (i == 0 ? "; font-weight: bold; width: 130px" : "") + "\">" + inlineMarkdownToHtml(cells[i], baseDir) + "</td>";
                }
                html += "</tr>";
                header = false;
            }
            html += "</table>";
            return html;
        }

        std::string displayBlockHtml(const std::string& tex) {
            return std::string("<table width=\"100%\" cellspacing=\"0\" cellpadding=\"12\" style=\"background: ") + kSurface +
                   "; border: 1px solid " + kDivider + "\"><tr><td align=\"left\">" + latexToHtml(tex, true) + "</td></tr></table>";
        }

        // Strips the front matter, returning the key/values found.
        std::map<std::string, std::string> stripFrontMatter(std::vector<std::string>& lines) {
            std::map<std::string, std::string> kv;
            if (lines.empty() || trim(lines[0]) != "---") return kv;
            std::size_t end = 1;
            while (end < lines.size() && trim(lines[end]) != "---") ++end;
            for (std::size_t i = 1; i < end && i < lines.size(); ++i) {
                const std::string& l = lines[i];
                const std::size_t colon = l.find(':');
                if (colon == std::string::npos) continue;
                kv[trim(l.substr(0, colon))] = trim(l.substr(colon + 1));
            }
            lines.erase(lines.begin(), lines.begin() + std::min(end + 1, lines.size()));
            return kv;
        }

        // Collects a $$ block starting at lines[i] (which begins with "$$");
        // returns the tex and advances i past the block.
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

    } // namespace

    std::string normalizeMathDelimiters(const std::string& md) {
        std::string out;
        out.reserve(md.size());
        bool inCode = false, inFence = false, inMath = false;
        for (std::size_t i = 0; i < md.size(); ++i) {
            const char c = md[i];
            if (!inCode && !inMath && md.compare(i, 3, "```") == 0) {
                inFence = !inFence;
                out += "```";
                i += 2;
                continue;
            }
            if (inFence) {
                out += c;
                continue;
            }
            if (c == '`' && !inMath) {
                inCode = !inCode;
                out += c;
                continue;
            }
            if (c == '$' && !inCode) {
                inMath = !inMath;
                out += c;
                continue;
            }
            if (c == '\\' && !inCode && !inMath && i + 1 < md.size()) {
                const char d = md[i + 1];
                if (d == '(' || d == '[') {
                    const std::string close = d == '(' ? "\\)" : "\\]";
                    const std::size_t end = md.find(close, i + 2);
                    if (end != std::string::npos) {
                        const std::string fence = d == '(' ? "$" : "$$";
                        out += fence + md.substr(i + 2, end - i - 2) + fence;
                        i = end + 1;
                        continue;
                    }
                }
            }
            out += c;
        }
        return out;
    }

    std::string helpMarkdownToHtml(const std::string& markdownIn, const std::string& baseDir) {
        const std::string markdown = normalizeMathDelimiters(markdownIn);
        std::vector<std::string> lines = splitLines(markdown);
        stripFrontMatter(lines);
        std::string html;
        std::size_t i = 0;
        while (i < lines.size()) {
            const std::string t = trim(lines[i]);
            if (t.empty()) {
                ++i;
                continue;
            }
            if (startsWith(t, "$$")) {
                html += displayBlockHtml(readDisplayBlock(lines, i));
                continue;
            }
            if (t[0] == '#') {
                std::size_t level = 0;
                while (level < t.size() && t[level] == '#') ++level;
                const std::string title = inlineMarkdownToHtml(trim(t.substr(level)), baseDir);
                // paragraphs rather than <h*>: QTextDocument keeps its own
                // heading sizes and ignores a font-size on them
                const int px = level <= 1 ? 20 : level == 2 ? 15
                                                            : 13;
                html += "<p style=\"font-size: " + std::to_string(px) +
                        "px; font-weight: 800; margin-top: 14px; margin-bottom: 4px\">" + title + "</p>";
                ++i;
                continue;
            }
            if (t[0] == '|') {
                std::vector<std::string> rows;
                while (i < lines.size() && !trim(lines[i]).empty() && trim(lines[i])[0] == '|') rows.push_back(lines[i++]);
                html += tableToHtml(rows, baseDir);
                continue;
            }
            const bool ul = startsWith(t, "- ") || startsWith(t, "* ");
            const bool ol = t.size() > 2 && std::isdigit(static_cast<unsigned char>(t[0])) && t.find(". ") != std::string::npos &&
                            t.find(". ") < 4;
            if (ul || ol) {
                html += ul ? "<ul style=\"margin-left: 14px\">" : "<ol style=\"margin-left: 14px\">";
                while (i < lines.size()) {
                    const std::string it = trim(lines[i]);
                    const bool isUl = startsWith(it, "- ") || startsWith(it, "* ");
                    const bool isOl = it.size() > 2 && std::isdigit(static_cast<unsigned char>(it[0])) && it.find(". ") != std::string::npos && it.find(". ") < 4;
                    if (!(isUl || isOl)) break;
                    const std::string body = isUl ? it.substr(2) : it.substr(it.find(". ") + 2);
                    html += "<li>" + inlineMarkdownToHtml(body, baseDir) + "</li>";
                    ++i;
                }
                html += ul ? "</ul>" : "</ol>";
                continue;
            }
            // paragraph
            std::string para;
            while (i < lines.size()) {
                const std::string pt = trim(lines[i]);
                if (pt.empty() || pt[0] == '#' || pt[0] == '|' || startsWith(pt, "$$") || startsWith(pt, "- ") || startsWith(pt, "* ")) break;
                if (!para.empty()) para += " ";
                para += pt;
                ++i;
            }
            html += "<p style=\"margin: 6px 0\">" + inlineMarkdownToHtml(para, baseDir) + "</p>";
        }
        return html;
    }

    // --- pages -----------------------------------------------------------------------

    HelpPage parseHelpMarkdown(const std::string& kind, const std::string& markdown) {
        HelpPage page;
        page.kind = kind;
        page.markdown = markdown;
        std::vector<std::string> lines = splitLines(markdown);
        const auto front = stripFrontMatter(lines);
        if (auto it = front.find("title"); it != front.end()) page.title = it->second;
        if (auto it = front.find("figure"); it != front.end()) page.figure = it->second;
        if (auto it = front.find("figure_path"); it != front.end()) page.figurePath = it->second;

        std::string section;   // lower-case current heading
        std::size_t i = 0;
        while (i < lines.size()) {
            const std::string t = trim(lines[i]);
            if (t.empty()) {
                ++i;
                continue;
            }
            if (startsWith(t, "$$")) {
                const std::string tex = readDisplayBlock(lines, i);
                if (page.tex.empty() && section.empty()) page.tex = tex;
                continue;
            }
            if (t[0] == '#') {
                std::size_t level = 0;
                while (level < t.size() && t[level] == '#') ++level;
                section = lower(trim(t.substr(level)));
                if (page.title.empty() && level == 1) page.title = trim(t.substr(level));
                ++i;
                continue;
            }
            if (t[0] == '|') {
                std::vector<std::string> rows;
                while (i < lines.size() && !trim(lines[i]).empty() && trim(lines[i])[0] == '|') rows.push_back(lines[i++]);
                if (section == "parameters") {
                    bool header = true;
                    for (const std::string& row : rows) {
                        if (isTableSeparator(row)) continue;
                        if (header) {
                            header = false;
                            continue;
                        }
                        const std::vector<std::string> cells = splitTableRow(row);
                        if (cells.empty()) continue;
                        HelpParam p;
                        std::string nameCell = cells[0];
                        std::size_t br = nameCell.find("<br");
                        if (br != std::string::npos) {
                            const std::size_t brEnd = nameCell.find('>', br);
                            p.range = trim(brEnd == std::string::npos ? std::string() : nameCell.substr(brEnd + 1));
                            nameCell = trim(nameCell.substr(0, br));
                        }
                        if (startsWith(nameCell, "**")) {
                            const std::size_t close = nameCell.find("**", 2);
                            p.name = close == std::string::npos ? nameCell.substr(2) : nameCell.substr(2, close - 2);
                        } else {
                            p.name = nameCell;
                        }
                        std::string body = cells.size() > 1 ? cells[1] : std::string();
                        const auto [open, close] = lastInlineMath(body);
                        if (open != std::string::npos && trim(body.substr(close + 1)).empty()) {
                            p.tex = trim(body.substr(open + 1, close - open - 1));
                            body = trim(body.substr(0, open));
                        }
                        p.body = body;
                        page.params.push_back(std::move(p));
                    }
                }
                continue;
            }
            std::string para;
            while (i < lines.size()) {
                const std::string pt = trim(lines[i]);
                if (pt.empty() || pt[0] == '#' || pt[0] == '|' || startsWith(pt, "$$")) break;
                if (!para.empty()) para += " ";
                para += pt;
                ++i;
            }
            if (section.empty()) {
                if (page.intro.empty()) page.intro = para;
            } else if (section == "note") {
                if (!page.note.empty()) page.note += "\n";
                page.note += para;
            }
        }
        if (page.title.empty()) page.title = kind;
        return page;
    }

    std::string helpDirectory(const std::string& hint) {
        auto usable = [](const std::string& dir) { return !dir.empty() && fs::is_directory(dir); };
        if (const char* env = std::getenv("SIRIUS_HELP_DIR"); env && usable(env)) return env;
        if (usable(hint)) return hint;
        // An installed tree before the checkout: an install on the machine it
        // was built on must still read its own pages, or a test of the
        // install proves nothing.
        if (std::string installed = installedDataDirectory("help"); !installed.empty()) return installed;
#ifdef SIRIUS_APP_SOURCE_DIR
        {
            // the checkout before the build tree's copy, so "Edit page" in a
            // development build changes the file under version control
            const std::string src = std::string(SIRIUS_APP_SOURCE_DIR) + "/help";
            if (usable(src)) return src;
        }
#endif
        // a build tree moved away from its checkout (or a checkout deleted)
        if (std::string beside = besideApplication("help"); !beside.empty()) return beside;
        if (!hint.empty()) return hint;
        return "help";
    }

    namespace {
        std::map<std::string, std::string>& memoryPages() {
            static std::map<std::string, std::string> pages;
            return pages;
        }
    } // namespace

    void registerHelpPage(const std::string& kind, const std::string& markdown) { memoryPages()[kind] = markdown; }

    HelpPage loadHelpPage(const std::string& kind, const std::string& hint) {
        const fs::path dir = helpDirectory(hint);
        const fs::path file = dir / (kind + ".md");
        std::ifstream in(file);
        if (!in) {
            auto mem = memoryPages().find(kind);
            if (mem != memoryPages().end() && !mem->second.empty()) {
                HelpPage page = parseHelpMarkdown(kind, mem->second);
                page.path = file.string();   // "Edit page" creates the override here
                return page;
            }
            HelpPage page;
            page.kind = kind;
            page.title = kind;
            page.intro = "No help page yet — click Edit page to write one.";
            page.path = file.string();
            page.markdown = "---\ntitle: " + kind + "\n---\n\n" + page.intro + "\n";
            return page;
        }
        std::stringstream ss;
        ss << in.rdbuf();
        HelpPage page = parseHelpMarkdown(kind, ss.str());
        page.path = file.string();
        if (!page.figurePath.empty()) {
            const fs::path fig(page.figurePath);
            if (!fig.is_absolute()) page.figurePath = (dir / fig).string();
        } else {
            for (const char* ext : {".png", ".svg", ".jpg", ".jpeg"}) {
                const fs::path candidate = dir / (kind + ext);
                if (fs::exists(candidate)) {
                    page.figurePath = candidate.string();
                    break;
                }
            }
        }
        return page;
    }

} // namespace sirius::app
