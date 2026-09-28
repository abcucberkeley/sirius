#include "imgui/panels/log_panel.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <string>
#include <utility>

#include <imgui.h>

#include "imgui/app.hpp"
#include "imgui/panels/diagnostic_cells.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

// The session log as a dock: everything Workbench::logLine records -- refused
// edits during a run, plugin errors, worker output, "no flagged labels" --
// where before only the last line flashed in the status bar for four seconds.
//
// Monospace, selectable, copy and clear, and an auto-scroll that stops
// following as soon as the reader scrolls up (a running step logs steadily,
// and a log that jumps away mid-read is unreadable).
//
// The view keeps its own copy of the lines: "Clear" empties the view while
// the session's own log is kept.

namespace sirius::app::gui {

    using theme::px;

    namespace {
        constexpr std::size_t kMaxLines = 5000;   // the workbench keeps the same number of lines
        constexpr float kMonoSize = 12.0f;
        constexpr double kCopiedSeconds = 1.2;

        // A position in the text: line and byte offset in it.
        struct TextPos {
            std::size_t line = 0, col = 0;
            bool operator<(const TextPos& o) const noexcept { return line < o.line || (line == o.line && col < o.col); }
            bool operator==(const TextPos& o) const noexcept { return line == o.line && col == o.col; }
        };
    } // namespace

    struct LogPanel::Impl {
        App& app;
        std::deque<std::string> lines;
        std::deque<float> widths;           // display width of each line at the last measured scale
        float widest = 0.0f;
        float measuredScale = 0.0f;
        int loggedConn = 0;

        bool following = true;
        bool scrollToBottom = true;         // the next frame puts the view on the last line
        int ignoreScrollFrames = 0;         // a programmatic scroll is not the reader scrolling
        float lastScrollY = 0.0f;
        double copiedUntil = 0.0;           // "Copied" on the button until then

        // selection: anchor where the drag started, caret where it is now
        bool hasSelection = false;
        bool selecting = false;
        TextPos anchor, caret;

        explicit Impl(App& a) : app(a) {}

        float measure(const std::string& s) const {
            ImFont* f = theme::mono() ? theme::mono() : ImGui::GetFont();
            const ImGuiStyle& st = ImGui::GetStyle();
            return f->CalcTextSizeA(kMonoSize * st.FontScaleMain * st.FontScaleDpi, FLT_MAX, 0.0f, s.c_str(), s.c_str() + s.size()).x;
        }

        void push(const std::string& line) {
            lines.push_back(line);
            widths.push_back(-1.0f);   // measured when next drawn (fonts may not be loaded yet)
            while (lines.size() > kMaxLines) {
                lines.pop_front();
                widths.pop_front();
                // the selection moves with the text it covers
                if (hasSelection) {
                    if (anchor.line == 0 || caret.line == 0) hasSelection = selecting = false;
                    else {
                        --anchor.line;
                        --caret.line;
                    }
                }
            }
        }

        void append(const std::string& line) {
            push(line);
            if (following) scrollToBottom = true;
        }

        void reload() {
            lines.clear();
            widths.clear();
            widest = 0.0f;
            hasSelection = selecting = false;
            for (const std::string& line : app.wb().log()) push(line);
            following = true;
            scrollToBottom = true;
        }

        void clear() {
            lines.clear();
            widths.clear();
            widest = 0.0f;
            hasSelection = selecting = false;
            following = true;
            scrollToBottom = true;
        }

        std::string selectedText() const {
            if (!hasSelection || lines.empty()) return {};
            TextPos a = anchor, b = caret;
            if (b < a) std::swap(a, b);
            std::string out;
            for (std::size_t i = a.line; i <= b.line && i < lines.size(); ++i) {
                const std::string& s = lines[i];
                const std::size_t from = i == a.line ? std::min(a.col, s.size()) : 0;
                const std::size_t to = i == b.line ? std::min(b.col, s.size()) : s.size();
                if (i != a.line) out += '\n';
                if (to > from) out.append(s, from, to - from);
            }
            return out;
        }

        std::string allText() const {
            std::string out;
            for (std::size_t i = 0; i < lines.size(); ++i) {
                if (i) out += '\n';
                out += lines[i];
            }
            return out;
        }

        // The byte offset in `s` closest to `x` display pixels from its start.
        std::size_t columnAt(const std::string& s, float x) const {
            if (x <= 0.0f) return 0;
            float prevW = 0.0f;
            std::size_t prev = 0;
            for (std::size_t i = 0; i < s.size();) {
                nextCodepoint(s, i);
                const float w = measureRange(s, i);
                if (w >= x) return (x - prevW) < (w - x) ? prev : i;
                prev = i;
                prevW = w;
            }
            return s.size();
        }

        float measureRange(const std::string& s, std::size_t end) const {
            ImFont* f = theme::mono() ? theme::mono() : ImGui::GetFont();
            const ImGuiStyle& st = ImGui::GetStyle();
            return f->CalcTextSizeA(kMonoSize * st.FontScaleMain * st.FontScaleDpi, FLT_MAX, 0.0f, s.c_str(), s.c_str() + end).x;
        }

        void drawHeader();
        void drawView();
    };

    void LogPanel::Impl::drawHeader() {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        const float h = theme::snap(px(theme::kDiagnosticsHeaderH));
        const float left = origin.x + px(14), right = origin.x + w - px(14);
        const float spacing = px(10);

        // right to left: Clear, Copy, and "Jump to latest" while not following
        const bool copied = ImGui::GetTime() < copiedUntil;
        const char* copyLabel = copied ? "Copied###copy" : "Copy###copy";
        auto buttonW = [](const char* shown, float fontSize, float padX) {
            return theme::snap(theme::textSize(shown, fontSize, theme::Weight::SemiBold).x + 2 * px(padX) + 2 * px(theme::kBorder));
        };
        const float clearW = buttonW("Clear", 12, 8);
        const float copyW = buttonW(copied ? "Copied" : "Copy", 12, 10);
        const float buttonH = theme::snap(std::max(px(14), theme::textSize("Copy", 12, theme::Weight::SemiBold).y) + 2 * px(4) + 2 * px(theme::kBorder));
        const float by = theme::snap(origin.y + (h - buttonH) * 0.5f);
        float x = right - clearW;
        ImGui::SetCursorScreenPos(ImVec2(x, by));
        {
            widgets::ButtonOpts o;
            o.kind = widgets::ButtonKind::Ghost;
            o.small = true;
            o.tooltip = "Empty this view; the session's own log is kept";
            if (widgets::button("Clear", o)) clear();
        }
        x -= spacing + copyW;
        ImGui::SetCursorScreenPos(ImVec2(x, by));
        {
            widgets::ButtonOpts o;
            o.small = true;
            o.tooltip = "Copy the whole log (or the selection) to the clipboard";
            if (widgets::button(copyLabel, o)) {
                const std::string text = hasSelection && !(anchor == caret) ? selectedText() : allText();
                ImGui::SetClipboardText(text.c_str());
                copiedUntil = ImGui::GetTime() + kCopiedSeconds;
            }
        }
        if (copied) app.requestRedraw();   // the label goes back to "Copy" by itself
        float restRight = x - spacing;
        if (!following) {
            const char* jump = "Jump to latest";
            const float jw = theme::textSize(jump, 11).x;
            const float lh = theme::textSize(jump, 11).y;
            x -= spacing + jw;
            ImGui::SetCursorScreenPos(ImVec2(x, theme::snap(origin.y + (h - lh) * 0.5f)));
            widgets::ButtonOpts o;
            o.kind = widgets::ButtonKind::Link;
            o.tooltip = "Scroll to the newest line and keep following it";
            if (widgets::button(jump, o)) {
                following = true;
                scrollToBottom = true;
            }
            restRight = x - spacing;
        }

        // left: the caption and the count
        const std::string cap = captionCase("Log · this session");
        float lx = left;
        if (restRight - lx > px(20)) {
            ImFont* cf = theme::captionFont() ? theme::captionFont() : ImGui::GetFont();
            const float cpx = cells::fontPx(theme::kCaptionPx);
            const std::string shownCap = cells::elideIn(cf, theme::kCaptionPx, cap, restRight - lx);
            const ImVec2 cs = cf->CalcTextSizeA(cpx, FLT_MAX, 0.0f, shownCap.c_str(), shownCap.c_str() + shownCap.size());
            dl->AddText(cf, cpx, ImVec2(theme::snap(lx), theme::snap(origin.y + (h - cs.y) * 0.5f)), theme::kNeutral600, shownCap.c_str(),
                        shownCap.c_str() + shownCap.size());
            lx += cs.x + spacing;
        }
        const std::size_t n = lines.size();
        const std::string count = format("%zu %s", n, n == 1 ? "line" : "lines");
        if (restRight - lx > theme::textSize(count, 11).x)
            widgets::drawTextIn(dl, ImVec2(lx, origin.y), ImVec2(restRight, origin.y + h), count, 11, theme::kNeutral600,
                                theme::Weight::Regular, 0.0f, 0.5f);

        // the hairline under the header
        ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + h));
        widgets::rule(theme::kHairline);
    }

    void LogPanel::Impl::drawView() {
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        if (avail.x < 2.0f || avail.y < 2.0f) return;
        ImFont* font = theme::mono() ? theme::mono() : ImGui::GetFont();
        const float size = kMonoSize * ImGui::GetStyle().FontScaleMain * ImGui::GetStyle().FontScaleDpi;
        const float lineH = std::ceil(size * 1.3f);
        const float padX = px(8), padY = px(6);

        // measure what is new (or everything, after a change of scale)
        const float scale = theme::scale();
        if (scale != measuredScale) {
            measuredScale = scale;
            widest = 0.0f;
            std::fill(widths.begin(), widths.end(), -1.0f);
        }
        for (std::size_t i = 0; i < lines.size(); ++i)
            if (widths[i] < 0.0f) {
                widths[i] = measure(lines[i]);
                widest = std::max(widest, widths[i]);
            }

        // the text box: surface fill, 1 px neutral-300 border
        ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kSurface);
        ImGui::PushStyleColor(ImGuiCol_Border, theme::kNeutral300);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, theme::crispPen(1));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        const bool open = ImGui::BeginChild("##logview", avail, ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
        if (open) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 base = ImGui::GetCursorScreenPos();   // scrolled origin of the content
            const float contentW = widest + 2 * padX;
            const float contentH = static_cast<float>(lines.size()) * lineH + 2 * padY;

            // The reader scrolled: following stops the moment they scroll up,
            // and resumes when they come back to the bottom themselves.
            const float scrollY = ImGui::GetScrollY(), maxY = ImGui::GetScrollMaxY();
            if (ignoreScrollFrames > 0) {
                --ignoreScrollFrames;
            } else if (std::abs(scrollY - lastScrollY) > 0.5f) {
                const bool bottom = scrollY >= maxY - 2.0f;
                if (bottom != following) following = bottom;
            }
            lastScrollY = scrollY;

            // the whole content area takes the mouse: click-drag selects
            const ImVec2 winMin = ImGui::GetWindowPos();
            const ImVec2 winSize = ImGui::GetWindowSize();
            ImGui::SetCursorScreenPos(base);
            ImGui::InvisibleButton("##text", ImVec2(std::max(contentW, winSize.x), std::max(contentH, 1.0f)));
            const bool hovered = ImGui::IsItemHovered();
            if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);
            auto posAt = [&](ImVec2 mouse) {
                TextPos p;
                if (lines.empty()) return p;
                const float fy = (mouse.y - base.y - padY) / lineH;
                if (fy < 0.0f) return p;
                p.line = std::min(lines.size() - 1, static_cast<std::size_t>(fy));
                if (fy >= static_cast<float>(lines.size())) {
                    p.col = lines.back().size();
                    return p;
                }
                p.col = columnAt(lines[p.line], mouse.x - base.x - padX);
                return p;
            };
            const ImGuiIO& io = ImGui::GetIO();
            if (ImGui::IsItemActivated()) {
                const TextPos p = posAt(io.MousePos);
                if (io.KeyShift && hasSelection) {
                    caret = p;
                } else {
                    anchor = caret = p;
                    hasSelection = true;
                }
                selecting = true;
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && p.line < lines.size()) {
                    // a double click takes the whole line
                    anchor = TextPos{p.line, 0};
                    caret = TextPos{p.line, lines[p.line].size()};
                    selecting = false;
                }
            }
            if (selecting && ImGui::IsItemActive()) {
                caret = posAt(io.MousePos);
                // dragging past the edge scrolls
                if (io.MousePos.y < winMin.y) ImGui::SetScrollY(ImGui::GetScrollY() - lineH);
                else if (io.MousePos.y > winMin.y + winSize.y) ImGui::SetScrollY(ImGui::GetScrollY() + lineH);
                app.requestRedraw();
            }
            if (!ImGui::IsItemActive()) selecting = false;
            // Ctrl+A / Ctrl+C while the log has the keyboard
            if (ImGui::IsWindowFocused() && !io.WantTextInput) {
                // Edit > Copy parameters is Ctrl+C too: the log's copy wins while it has focus
                app.claimKey(ImGuiMod_Ctrl | ImGuiKey_A);
                app.claimKey(ImGuiMod_Ctrl | ImGuiKey_C);
                if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_A) && !lines.empty()) {
                    anchor = TextPos{0, 0};
                    caret = TextPos{lines.size() - 1, lines.back().size()};
                    hasSelection = true;
                }
                if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_C) && hasSelection && !(anchor == caret)) {
                    const std::string text = selectedText();
                    ImGui::SetClipboardText(text.c_str());
                }
            }

            // only the lines on screen are drawn
            const float top = winMin.y, bottom = winMin.y + winSize.y;
            std::size_t first = 0, last = 0;
            if (!lines.empty()) {
                const float f0 = (top - base.y - padY) / lineH;
                first = f0 <= 0.0f ? 0 : std::min(lines.size(), static_cast<std::size_t>(f0));
                const float f1 = (bottom - base.y - padY) / lineH + 1.0f;
                last = f1 <= 0.0f ? 0 : std::min(lines.size(), static_cast<std::size_t>(f1));
            }
            TextPos a = anchor, b = caret;
            if (b < a) std::swap(a, b);
            const bool showSel = hasSelection && !(a == b);
            for (std::size_t i = first; i < last; ++i) {
                const std::string& s = lines[i];
                const ImVec2 at(base.x + padX, base.y + padY + static_cast<float>(i) * lineH);
                const float ty = theme::snap(at.y + (lineH - size) * 0.5f);
                dl->AddText(font, size, ImVec2(theme::snap(at.x), ty), theme::kNeutral700, s.c_str(), s.c_str() + s.size());
                if (showSel && i >= a.line && i <= b.line) {
                    // selected text: paper on the accent
                    const std::size_t from = i == a.line ? std::min(a.col, s.size()) : 0;
                    const std::size_t to = i == b.line ? std::min(b.col, s.size()) : s.size();
                    const float x0 = at.x + measureRange(s, from);
                    float x1 = at.x + measureRange(s, to);
                    if (i != b.line) x1 += size * 0.4f;   // the line break is part of it
                    if (x1 > x0) {
                        dl->AddRectFilled(ImVec2(x0, at.y), ImVec2(x1, at.y + lineH), theme::kAccent);
                        if (to > from)
                            dl->AddText(font, size, ImVec2(theme::snap(x0), ty), theme::kBg, s.c_str() + from, s.c_str() + to);
                    }
                }
            }

            if (scrollToBottom) {
                // past the end: Dear ImGui clamps it to the new bottom next frame
                ImGui::SetScrollY(contentH);
                ImGui::SetScrollX(0.0f);
                scrollToBottom = false;
                ignoreScrollFrames = 2;
                app.requestRedraw();
            }
        }
        ImGui::EndChild();
    }

    LogPanel::LogPanel(App& app) : impl_(std::make_unique<Impl>(app)) {
        Impl& d = *impl_;
        d.reload();
        Impl* raw = impl_.get();
        d.loggedConn = app.bridge().logged.connect([raw](const std::string& line) { raw->append(line); });
    }

    LogPanel::~LogPanel() { impl_->app.bridge().logged.disconnect(impl_->loggedConn); }

    void LogPanel::draw() {
        Impl& d = *impl_;
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        ImGui::GetWindowDrawList()->AddRectFilled(origin, ImVec2(origin.x + avail.x, origin.y + avail.y), theme::kBg);
        ImGui::PushID("log");
        d.drawHeader();
        d.drawView();
        ImGui::PopID();
    }

    void LogPanel::showLatest() {
        impl_->following = true;
        impl_->scrollToBottom = true;
        impl_->app.requestRedraw();
    }

    int LogPanel::lineCount() const { return static_cast<int>(impl_->lines.size()); }

} // namespace sirius::app::gui
