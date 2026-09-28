#include "imgui/panels/help_window.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#include <imgui.h>

#include "core/help_pages.hpp"
#include "imgui/app.hpp"
#include "imgui/platform.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/panels/markdown_view.hpp"
#include "imgui/widgets/controls.hpp"
#include "imgui/widgets/icons.hpp"

namespace sirius::app::gui {

    namespace fs = std::filesystem;
    using theme::px;
    using theme::Weight;

    namespace {

        constexpr float kWidth = 520;
        constexpr float kHeight = 700;
        constexpr float kMaxHeight = 760;
        constexpr float kMinWidth = 360;
        constexpr float kHeaderH = 36;
        constexpr float kMargin = 22;          // the page's margin
        constexpr float kFigureH = 170;        // the dashed drop zone
        constexpr double kPollSeconds = 1.0;   // how often the page file is looked at
        constexpr double kSettleSeconds = 0.2; // an editor may write the file in pieces

        // Markdown of the page after the parts the structured layout renders
        // itself (front matter, intro, first display formula), split into the
        // sections before and after "## Parameters" (which, like "## Note",
        // is rendered from the parsed page), so further sections keep their
        // place.
        struct Remainder {
            std::string before, after;
        };
        Remainder remainderMarkdown(const HelpPage& page) {
            std::istringstream in(page.markdown);
            std::string line;
            Remainder out;
            bool inFront = false, frontDone = false, introDone = page.intro.empty(), texDone = page.tex.empty();
            bool inTex = false, inIntro = false, afterParams = false, skipping = false;
            std::size_t lineNo = 0;
            while (std::getline(in, line)) {
                const std::string t = trimmed(line);
                ++lineNo;
                if (!frontDone) {
                    if (lineNo == 1 && t == "---") {
                        inFront = true;
                        continue;
                    }
                    if (inFront) {
                        if (t == "---") {
                            inFront = false;
                            frontDone = true;
                        }
                        continue;
                    }
                    frontDone = true;
                }
                if (inTex) {
                    if (t.find("$$") != std::string::npos) {
                        inTex = false;
                        texDone = true;
                    }
                    continue;
                }
                if (!texDone && startsWith(t, "$$")) {
                    if (t.find("$$", 2) != std::string::npos) texDone = true;
                    else inTex = true;
                    continue;
                }
                if (inIntro) {
                    if (t.empty()) {
                        inIntro = false;
                        introDone = true;
                    }
                    continue;
                }
                if (!introDone && !t.empty() && t[0] != '#' && t[0] != '|') {
                    inIntro = true;
                    continue;
                }
                if (!t.empty() && t[0] == '#') {
                    std::size_t level = 0;
                    while (level < t.size() && t[level] == '#') ++level;
                    const std::string name = toLower(trimmed(t.substr(level)));
                    skipping = name == "parameters" || name == "note";
                    if (name == "parameters") afterParams = true;
                }
                if (skipping) continue;
                (afterParams ? out.after : out.before) += line + "\n";
            }
            return out;
        }

        bool hasText(const std::string& s) { return s.find_first_not_of(" \t\r\n") != std::string::npos; }

        // --- the shortcuts page in this platform's notation -----------------------

#ifndef __APPLE__
        // The page writes shortcuts the way the Mac menus do (⇧⌘E). Here a
        // modifier glyph glued to a key becomes the notation the menus of
        // this application use (Ctrl+Shift+E); a glyph on its own -- in
        // "⌘ stands for Ctrl" or the formula that explains the glyphs --
        // is left as written.
        std::string platformKeys(const std::string& md) {
            constexpr char32_t kCmd = 0x2318, kOpt = 0x2325, kShift = 0x21E7, kBackspace = 0x232B;
            std::string out;
            out.reserve(md.size());
            for (std::size_t i = 0; i < md.size();) {
                const std::size_t at = i;
                const char32_t c = nextCodepoint(md, i);
                if (c == kBackspace) {
                    out += "Backspace";
                    continue;
                }
                if (c != kCmd && c != kOpt && c != kShift) {
                    out.append(md, at, i - at);
                    continue;
                }
                bool ctrl = false, alt = false, shift = false;
                std::size_t j = at;   // past the modifiers
                while (j < md.size()) {
                    std::size_t k = j;
                    const char32_t m = nextCodepoint(md, k);
                    if (m == kCmd) ctrl = true;
                    else if (m == kOpt) alt = true;
                    else if (m == kShift) shift = true;
                    else break;
                    j = k;
                }
                const std::size_t keyAt = j;
                char32_t key = 0;
                if (j < md.size()) key = nextCodepoint(md, j);
                const bool glued = key != 0 && key != ' ' && key != '\t' && key != '}' && key != '\n' && key != '\r';
                if (!glued) {
                    out.append(md, at, keyAt - at);   // the glyphs as written
                    i = keyAt;
                    continue;
                }
                if (ctrl) out += "Ctrl+";
                if (alt) out += "Alt+";
                if (shift) out += "Shift+";
                switch (key) {
                    case 0x2191: out += "Up"; break;
                    case 0x2193: out += "Down"; break;
                    case 0x2190: out += "Left"; break;
                    case 0x2192: out += "Right"; break;
                    case kBackspace: out += "Backspace"; break;
                    default: out.append(md, keyAt, j - keyAt); break;
                }
                i = j;
            }
            return out;
        }
#endif

        // Shortcuts of this application the shared page does not list.
        std::string extraShortcuts() {
            struct Entry {
                ImGuiKeyChord keys;
                const char* action;
            };
            static const Entry entries[] = {
                {ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_O, "Open folder as dataset"},
                {ImGuiMod_Ctrl | ImGuiKey_Y, "Redo"},
                {ImGuiMod_Shift | ImGuiKey_R, "Reset contrast (display)"},
                {ImGuiMod_Ctrl | ImGuiKey_Backspace, "Delete label"},
                {ImGuiMod_Alt | ImGuiKey_6, "User operations"},
                {keys::logDock, "Log"},
            };
            std::string md = "\n## Also in this window\n\n| Shortcut | Action |\n|---|---|\n";
            for (const Entry& e : entries) md += "| " + shortcutText(e.keys) + " | " + e.action + " |\n";
            return md;
        }

        bool isFigureFile(const std::string& path) {
            const std::string ext = toLower(fs::u8path(path).extension().u8string());
            return ext == ".png" || ext == ".svg" || ext == ".jpg" || ext == ".jpeg" || ext == ".pdf";
        }

        std::string readText(const fs::path& path) {
            std::ifstream in(path, std::ios::binary);
            std::stringstream ss;
            ss << in.rdbuf();
            return ss.str();
        }

    } // namespace

    struct HelpWindow::Impl {
        App& app;
        explicit Impl(App& a) : app(a) {}

        bool visible = false;
        bool raise = false;
        bool placed = false;
        std::string kind;
        HelpPage page;
        std::string source;             // the page's Markdown as the file holds it
        bool followSelection = true;
        std::uint64_t seenSelection = 0;

        // the page file, looked at while the window shows
        bool fileExists = false;
        fs::file_time_type fileTime{};
        double nextPoll = 0.0;
        double reloadAt = -1.0;

        // the page as laid out for one width and one scale
        markdown::Images images;
        markdown::Layout top, bottom;   // above and below the figure
        float figureTop = 0.0f;         // where the figure starts, below `top`
        float layoutWidth = -1.0f, layoutScale = -1.0f;
        bool dirty = true;
        bool scrollToTop = false;

        void stamp() {
            std::error_code ec;
            fileExists = !page.path.empty() && fs::exists(page.path, ec);
            fileTime = fileExists ? fs::last_write_time(page.path, ec) : fs::file_time_type{};
        }

        void load(const std::string& k) {
            kind = k;
            page = loadHelpPage(k);
            source = page.markdown;
#ifndef __APPLE__
            if (k == "shortcuts") {
                // this application's notation, and what only it has
                HelpPage shown = parseHelpMarkdown(k, platformKeys(page.markdown) + extraShortcuts());
                shown.path = page.path;
                shown.figurePath = page.figurePath;
                page = std::move(shown);
            }
#else
            if (k == "shortcuts") {
                HelpPage shown = parseHelpMarkdown(k, page.markdown + extraShortcuts());
                shown.path = page.path;
                shown.figurePath = page.figurePath;
                page = std::move(shown);
            }
#endif
            images.clear();
            dirty = true;
            scrollToTop = true;
            reloadAt = -1.0;
            stamp();
            app.requestRedraw();
        }

        // --- the file ------------------------------------------------------------

        void poll() {
            const double now = ImGui::GetTime();
            if (reloadAt >= 0.0) {
                if (now >= reloadAt) load(kind);
                return;
            }
            if (now < nextPoll) return;
            nextPoll = now + kPollSeconds;
            if (page.path.empty()) return;
            std::error_code ec;
            const bool exists = fs::exists(page.path, ec);
            const fs::file_time_type t = exists ? fs::last_write_time(page.path, ec) : fs::file_time_type{};
            // a page that was never on disk is not watched; a page that goes
            // away keeps what is shown
            if (!exists || (exists == fileExists && t == fileTime)) return;
            reloadAt = now + kSettleSeconds;
            app.requestRedraw(12);
        }

        void editPage() {
            if (page.path.empty()) return;
            std::error_code ec;
            if (!fs::exists(page.path, ec)) {
                fs::create_directories(fs::path(page.path).parent_path(), ec);
                std::ofstream(page.path, std::ios::binary) << source;
                stamp();
            }
            platform::openInFileManager(page.path);
        }

        // Copies the image next to the page as <kind>-figure.<ext> and
        // references it from the front matter (an image dropped on the window).
        void setFigure(const std::string& file) {
            if (page.path.empty() || !isFigureFile(file)) return;
            const fs::path pagePath(page.path);
            const std::string ext = toLower(fs::u8path(file).extension().u8string()).substr(1);
            const fs::path target = pagePath.parent_path() / (kind + "-figure." + ext);
            std::error_code ec;
            fs::create_directories(pagePath.parent_path(), ec);
            fs::copy_file(fs::u8path(file), target, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                app.wb().logLine("Help: could not copy the figure: " + ec.message());
                return;
            }
            // the file as it is now (it may have been edited since it was read)
            std::string md = fs::exists(pagePath, ec) ? readText(pagePath) : source;
            const std::string key = "figure_path:";
            const std::size_t at = md.find(key);
            const std::string value = target.filename().string();
            if (at != std::string::npos) {
                const std::size_t eol = md.find('\n', at);
                md.replace(at, (eol == std::string::npos ? md.size() : eol) - at, key + " " + value);
            } else if (md.rfind("---\n", 0) == 0) {
                md.insert(4, key + " " + value + "\n");
            } else {
                md = "---\n" + key + " " + value + "\n---\n\n" + md;
            }
            std::ofstream(pagePath, std::ios::binary) << md;
            load(kind);
        }

        void chooseFigure() {
            app.defer([this] {
                const std::string file =
                    platform::openFileDialog("Choose figure", app.lastDir(), {{"Images", "png,svg,jpg,jpeg,pdf"}});
                if (file.empty()) return;
                app.setLastDir(parentPath(file));
                setFigure(file);
            });
        }

        // --- layout -------------------------------------------------------------------

        markdown::TextStyle body() const {
            markdown::TextStyle s;
            s.px = theme::kBodyPx;
            s.color = theme::kText;
            return s;
        }

        markdown::Context context() {
            markdown::Context c;
            c.baseDir = page.path.empty() ? std::string() : fs::path(page.path).parent_path().u8string();
            c.images = &images;
            return c;
        }

        // The display formula on the surface, 12 px inside a 1 px rule.
        markdown::Layout displayFormula(const std::string& tex, float width) const {
            const float pad = px(12);
            markdown::TextStyle s = body();
            s.px = theme::kMonoPx;   // latexToHtml sets display formulas at 15 px
            const markdown::Layout f = markdown::formula(tex, true, s, std::max(px(40), width - 2 * pad));
            markdown::Layout out;
            const float h = f.height + 2 * pad;
            out.fill(ImVec2(0, 0), ImVec2(width, h), theme::kSurface);
            out.frame(ImVec2(0, 0), ImVec2(width, h), theme::kDivider, 1.0f);
            out.place(f, pad, pad);
            out.width = width;
            out.height = h;
            return out;
        }

        void build(float width) {
            const markdown::Context ctx = context();
            const markdown::TextStyle text = body();
            const Remainder rest = remainderMarkdown(page);

            // title, intro, display formula
            top = markdown::Layout();
            float y = 0.0f;
            {
                markdown::TextStyle s = text;
                s.px = theme::kH3Px;
                s.weight = Weight::ExtraBold;
                s.leading = 1.25f;
                const markdown::Layout title = markdown::plain(page.title, s, width);
                top.place(title, 0.0f, y);
                y += title.height + px(10);
            }
            if (!page.intro.empty()) {
                const markdown::Layout intro = markdown::paragraph(page.intro, text, width, ctx);
                top.place(intro, 0.0f, std::floor(y));
                y += intro.height + px(14);
            }
            if (!page.tex.empty()) {
                const markdown::Layout f = displayFormula(page.tex, width);
                top.place(f, 0.0f, std::floor(y));
                y += f.height + px(14);
            }
            figureTop = std::floor(y);
            top.width = width;
            top.height = figureTop;

            // sections before the parameters, the parameters, the rest, the note
            bottom = markdown::Layout();
            y = 0.0f;
            if (hasText(rest.before)) {
                const markdown::Layout b = markdown::blocks(rest.before, text, width, ctx);
                bottom.place(b, 0.0f, y);
                y += b.height + px(12);
            }
            if (!page.params.empty()) {
                markdown::TextStyle cap = text;
                cap.px = theme::kCaptionPx;
                cap.face = markdown::Face::Caption;
                cap.color = theme::kNeutral600;
                const markdown::Layout head = markdown::plain("Parameters", cap, width);
                bottom.place(head, 0.0f, y);
                y = std::floor(y + head.height + px(6));
                const float rule = theme::crispPen(2.0f);
                bottom.fill(ImVec2(0.0f, y), ImVec2(width, y + rule), theme::kDivider);
                y += rule;
                const float nameW = px(120), gap = px(12), padY = px(10);
                const float textW = std::max(px(60), width - nameW);
                for (const HelpParam& p : page.params) {
                    markdown::TextStyle name = text;
                    name.px = 12;
                    name.weight = Weight::ExtraBold;
                    markdown::TextStyle range = text;
                    range.px = theme::kSmallPx;
                    range.color = theme::kNeutral600;
                    const markdown::Layout n = markdown::plain(p.name, name, nameW - gap);
                    const markdown::Layout r = markdown::plain(p.range, range, nameW - gap);
                    const float leftH = n.height + (p.range.empty() ? 0.0f : r.height);
                    const markdown::Layout b = markdown::paragraph(p.body, text, textW, ctx);
                    float rightH = b.height;
                    markdown::Layout f;
                    if (!p.tex.empty()) {
                        markdown::TextStyle ts = text;
                        ts.color = theme::kNeutral800;
                        f = markdown::formula(p.tex, p.tex.find("\\frac") != std::string::npos, ts, textW);
                        rightH += px(4) + f.height;
                    }
                    const float top0 = std::floor(y + padY);
                    bottom.place(n, 0.0f, top0);
                    if (!p.range.empty()) bottom.place(r, 0.0f, top0 + n.height);
                    bottom.place(b, nameW, top0);
                    if (!p.tex.empty()) bottom.place(f, nameW, std::floor(top0 + b.height + px(4)));
                    y = std::floor(top0 + std::max(leftH, rightH) + padY);
                    const float line = theme::crispPen(1.0f);
                    bottom.fill(ImVec2(0.0f, y), ImVec2(width, y + line), theme::kDivider);
                    y += line;
                }
            }
            if (hasText(rest.after)) {
                y += px(12);
                const markdown::Layout a = markdown::blocks(rest.after, text, width, ctx);
                bottom.place(a, 0.0f, std::floor(y));
                y += a.height;
            }
            if (!page.note.empty()) {
                y += px(14);
                markdown::TextStyle s = text;
                s.px = theme::kSmallPx;
                s.color = theme::kNeutral600;
                const markdown::Layout n = markdown::blocks(page.note, s, width, ctx);
                bottom.place(n, 0.0f, std::floor(y));
                y += n.height;
            }
            bottom.width = width;
            bottom.height = std::ceil(y);
            layoutWidth = width;
            layoutScale = theme::scale();
            dirty = false;
        }

        // --- drawing -------------------------------------------------------------------

        // The figure the page names, centred with its caption; without one
        // the dashed slot that asks for it. Drops cannot reach this window,
        // so the slot offers a file dialog doing what a drop did.
        void drawFigure(float width) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const bool hasFile = !page.figurePath.empty() && isFile(page.figurePath);
            std::shared_ptr<Texture> tex = hasFile ? images.get(page.figurePath) : nullptr;
            if (tex && tex->valid() && tex->width() > 0) {
                const float pad = px(6);
                float w = px(static_cast<float>(tex->width())), h = px(static_cast<float>(tex->height()));
                const float room = std::max(px(40), width - 2 * pad);
                if (w > room) {
                    h *= room / w;
                    w = room;
                }
                w = std::floor(w);
                h = std::floor(h);
                const float x = std::floor(at.x + (width - w) * 0.5f);
                dl->AddImage(tex->ref(), ImVec2(x, at.y + pad), ImVec2(x + w, at.y + pad + h));
                float y = at.y + pad + h;
                if (!page.figure.empty()) {
                    markdown::TextStyle s = body();
                    s.px = theme::kSmallPx;
                    s.color = theme::kNeutral600;
                    const markdown::Layout cap = markdown::plain(page.figure, s, width, markdown::Align::Center);
                    cap.draw(dl, ImVec2(at.x, y));
                    y += cap.height;
                }
                ImGui::Dummy(ImVec2(width, y - at.y));
                // another image in its place
                const float linkW = theme::textSize("Replace figure\xE2\x80\xA6", theme::kSmallPx).x;
                ImGui::SetCursorScreenPos(ImVec2(std::floor(at.x + (width - linkW) * 0.5f), y + px(2)));
                widgets::ButtonOpts o;
                o.kind = widgets::ButtonKind::Link;
                o.tooltip = "Copy an image next to the page and show it here";
                if (widgets::button("Replace figure\xE2\x80\xA6##figure", o)) chooseFigure();
                ImGui::SetCursorScreenPos(ImVec2(at.x, ImGui::GetItemRectMax().y + pad));
                ImGui::Dummy(ImVec2(width, 0.0f));
                return;
            }

            const float h = px(kFigureH);
            const ImVec2 max(at.x + width, at.y + h);
            widgets::dashedRect(dl, at, max, theme::kNeutral400, 1.0f);
            // "Figure", the caption, and the way to supply one
            markdown::TextStyle s = body();
            s.px = 12;
            s.color = theme::kNeutral600;
            markdown::TextStyle bold = s;
            bold.weight = Weight::ExtraBold;
            const float inner = std::max(px(40), width - px(20));
            const markdown::Layout l1 = markdown::plain("Figure", bold, inner, markdown::Align::Center);
            const markdown::Layout l2 = markdown::plain(page.figure, s, inner, markdown::Align::Center);
            std::string why;
            if (hasFile) why = fileName(page.figurePath) + " cannot be shown here";
            markdown::Layout l3;
            if (!why.empty()) l3 = markdown::plain(why, s, inner, markdown::Align::Center);
            const float linkH = theme::textSize("Choose figure", 12).y;
            const float total = l1.height + l2.height + l3.height + linkH;
            float y = std::floor(at.y + (h - total) * 0.5f);
            const float x = at.x + px(10);
            l1.draw(dl, ImVec2(x, y));
            y += l1.height;
            l2.draw(dl, ImVec2(x, y));
            y += l2.height;
            if (!why.empty()) {
                l3.draw(dl, ImVec2(x, y));
                y += l3.height;
            }
            const std::string label = "Choose figure\xE2\x80\xA6 \xC2\xB7 PNG, SVG, PDF";
            const float linkW = theme::textSize(label, theme::kSmallPx).x;
            ImGui::SetCursorScreenPos(ImVec2(std::floor(at.x + (width - linkW) * 0.5f), y));
            widgets::ButtonOpts o;
            o.kind = widgets::ButtonKind::Link;
            o.enabled = !page.path.empty();
            o.tooltip = "Copy an image next to the page as its figure (dropping a file here is not possible in this window)";
            if (widgets::button((label + "##figure").c_str(), o)) chooseFigure();
            ImGui::SetCursorScreenPos(at);
            ImGui::Dummy(ImVec2(width, h));
        }

        void drawHeader(float width) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 min = ImGui::GetCursorScreenPos();
            const float h = theme::snap(px(kHeaderH));
            const ImVec2 max(min.x + width, min.y + h);

            // the header moves the window (it has no title bar)
            ImGui::SetNextItemAllowOverlap();
            ImGui::InvisibleButton("##helpHeader", ImVec2(width, h));
            if (ImGui::IsItemHovered() || ImGui::IsItemActive()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
                const ImVec2 d = ImGui::GetIO().MouseDelta;
                const ImVec2 p = ImGui::GetWindowPos();
                ImGui::SetWindowPos(ImVec2(p.x + d.x, p.y + d.y));
            }

            // right to left: close, Edit page
            const float margin = px(14), spacing = px(12);
            const float closeS = theme::snap(px(18));
            float right = max.x - margin;
            ImGui::SetCursorScreenPos(ImVec2(right - closeS, std::floor(min.y + (h - closeS) * 0.5f)));
            widgets::GlyphOpts g;
            g.borderless = true;
            g.iconPx = 11;
            g.tooltip = "Close";
            if (widgets::glyphButton("##closeHelp", Icon::Close, 18, g)) visible = false;
            right -= closeS + spacing;

            const char* edit = "Edit page";
            const ImVec2 editSize = theme::textSize(edit, theme::kSmallPx);
            ImGui::SetCursorScreenPos(ImVec2(right - editSize.x, std::floor(min.y + (h - editSize.y) * 0.5f)));
            widgets::ButtonOpts o;
            o.kind = widgets::ButtonKind::Link;
            o.enabled = !page.path.empty();
            o.tooltip = "Open the Markdown page in your editor; the window reloads when the file changes";
            if (widgets::button(edit, o)) editPage();
            right -= editSize.x + spacing;

            // "HELP · <TITLE>"
            const std::string caption = "HELP \xC2\xB7 " + captionCase(page.title);
            const float room = std::max(0.0f, right - (min.x + margin));
            const std::string shown = widgets::elideText(caption, room, theme::kCaptionPx);
            {
                const theme::FontScope f(theme::kCaptionPx, theme::captionFont());
                const ImVec2 ts = ImGui::CalcTextSize(shown.c_str());
                dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(min.x + margin, std::floor(min.y + (h - ts.y) * 0.5f)),
                            theme::kNeutral600, shown.c_str());
            }

            // the 2 px rule under the header
            const float rule = theme::crispPen(theme::kRule);
            dl->AddRectFilled(ImVec2(min.x, max.y), ImVec2(max.x, max.y + rule), theme::kDivider);
            ImGui::SetCursorScreenPos(ImVec2(min.x, max.y + rule));
            ImGui::Dummy(ImVec2(width, 0.0f));
            ImGui::SetCursorScreenPos(ImVec2(min.x, max.y + rule));
        }

        void drawBody() {
            const float margin = px(kMargin);
            ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kBg);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            const float border = theme::crispPen(2.0f);
            if (ImGui::BeginChild("##helpBody", ImVec2(avail.x - border, avail.y - border), ImGuiChildFlags_None)) {
                if (scrollToTop) {
                    ImGui::SetScrollY(0.0f);
                    scrollToTop = false;
                }
                const float width = std::floor(ImGui::GetContentRegionAvail().x - 2 * margin);
                if (width > px(40)) {
                    if (dirty || std::abs(width - layoutWidth) > 0.5f || layoutScale != theme::scale()) build(width);
                    const float x0 = ImGui::GetCursorPosX() + margin;
                    ImGui::SetCursorPos(ImVec2(x0, ImGui::GetCursorPosY() + margin));
                    markdown::show(top);
                    ImGui::SetCursorPosX(x0);
                    drawFigure(width);
                    ImGui::SetCursorPosX(x0);
                    widgets::vspace(14);
                    ImGui::SetCursorPosX(x0);
                    markdown::show(bottom);
                    widgets::vspace(kMargin);
                }
            }
            ImGui::EndChild();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();
        }
    };

    HelpWindow::HelpWindow(App& app) : impl_(std::make_unique<Impl>(app)) {
        impl_->seenSelection = app.bridge().rev().selection;
        impl_->load("manual");
    }

    HelpWindow::~HelpWindow() = default;

    void HelpWindow::draw() {
        Impl& d = *impl_;
        d.seenSelection = d.app.bridge().rev().selection;
        if (!d.visible) return;
        d.poll();

        const ImGuiViewport* vp = ImGui::GetMainViewport();
        if (!d.placed) {
            // over the right part of the viewer, clear of the parameters
            const float w = px(kWidth), h = std::min(px(kHeight), vp->WorkSize.y - px(80));
            const float x = std::max(vp->WorkPos.x + px(20), vp->WorkPos.x + vp->WorkSize.x - px(theme::kParamsDockW) - w - px(24));
            ImGui::SetNextWindowPos(ImVec2(x, vp->WorkPos.y + px(64)));
            ImGui::SetNextWindowSize(ImVec2(w, h));
            d.placed = true;
        }
        ImGui::SetNextWindowSizeConstraints(ImVec2(px(kMinWidth), px(200)), ImVec2(FLT_MAX, px(kMaxHeight)));
        if (d.raise) ImGui::SetNextWindowFocus();
        d.raise = false;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBg);
        ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, theme::crispPen(2.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
        const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking |
                                       ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                                       ImGuiWindowFlags_NoScrollWithMouse;
        const bool open = ImGui::Begin("Help###siriusHelpWindow", nullptr, flags);
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(2);
        if (open) {
            // inside the 2 px ink border
            const float border = theme::crispPen(2.0f);
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            ImGui::SetCursorScreenPos(ImVec2(origin.x + border, origin.y + border));
            const float width = ImGui::GetContentRegionAvail().x - border;
            d.drawHeader(width);
            d.drawBody();
        }
        ImGui::End();
    }

    void HelpWindow::showKind(const std::string& kind) {
        Impl& d = *impl_;
        // The application calls this again whenever the selection moves while
        // the window is open. A page the user asked for by name (the manual,
        // the shortcuts) stays put then.
        const std::uint64_t sel = d.app.bridge().rev().selection;
        const bool followCall = d.visible && sel != d.seenSelection;
        d.seenSelection = sel;
        if (followCall && !d.followSelection) return;
        d.followSelection = true;
        d.load(kind);
        d.visible = true;
        d.raise = true;
    }

    void HelpWindow::showManual() {
        Impl& d = *impl_;
        d.followSelection = false;
        d.load("manual");
        d.visible = true;
        d.raise = true;
    }

    void HelpWindow::showShortcuts() {
        Impl& d = *impl_;
        d.followSelection = false;
        d.load("shortcuts");
        d.visible = true;
        d.raise = true;
    }

    std::string HelpWindow::currentKind() const { return impl_->kind; }
    bool HelpWindow::visible() const { return impl_->visible; }

    void HelpWindow::setVisible(bool on) {
        Impl& d = *impl_;
        if (on && !d.visible) d.raise = true;
        d.visible = on;
        d.app.requestRedraw();
    }

} // namespace sirius::app::gui
