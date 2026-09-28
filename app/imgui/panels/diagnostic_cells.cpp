#include "imgui/panels/diagnostic_cells.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <imgui.h>
#include <implot.h>

#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    using theme::px;
    using theme::Weight;

    namespace {

        std::string formatNumber(double v) {
            if (!std::isfinite(v)) return "–";
            if (std::abs(v) >= 1000.0 || v == std::floor(v)) return format("%lld", static_cast<long long>(std::llround(v)));
            return format("%.4g", v);
        }

        ImFont* bodyFont() {
            ImFont* f = theme::font();
            return f ? f : ImGui::GetFont();
        }

        // `text` broken at `wrap` display pixels, every line centred in `r`.
        void drawWrappedCentered(ImDrawList* dl, const cells::Rect& r, const std::string& text, float designPx, ImU32 color) {
            if (text.empty() || r.width() <= 1.0f || r.height() <= 1.0f) return;
            ImFont* f = bodyFont();
            const float size = cells::fontPx(designPx);
            const char* p = text.c_str();
            const char* const end = p + text.size();
            std::vector<std::pair<const char*, const char*>> lines;
            while (p < end) {
                const char* e = f->CalcWordWrapPosition(size, p, end, r.width());
                if (e <= p) {
                    // not even one character fits: take it anyway, or this never ends
                    std::size_t i = static_cast<std::size_t>(p - text.c_str());
                    nextCodepoint(text, i);
                    e = text.c_str() + i;
                }
                lines.emplace_back(p, e);
                p = e;
                while (p < end && (*p == ' ' || *p == '\n')) ++p;
            }
            const float lineH = std::ceil(size * 1.25f);
            float y = r.min.y + (r.height() - lineH * static_cast<float>(lines.size())) * 0.5f;
            dl->PushClipRect(r.min, r.max, true);
            for (const auto& line : lines) {
                const float w = f->CalcTextSizeA(size, FLT_MAX, 0.0f, line.first, line.second).x;
                dl->AddText(f, size, ImVec2(theme::snap(r.min.x + (r.width() - w) * 0.5f), theme::snap(y)), color, line.first, line.second);
                y += lineH;
            }
            dl->PopClipRect();
        }

        // Height of `text` wrapped at `wrap` display pixels in the body face.
        float wrappedHeight(const std::string& text, float designPx, float wrap) {
            if (text.empty()) return 0.0f;
            return bodyFont()->CalcTextSizeA(cells::fontPx(designPx), FLT_MAX, std::max(1.0f, wrap), text.c_str(), text.c_str() + text.size()).y;
        }

        void drawWrapped(ImDrawList* dl, ImVec2 pos, const std::string& text, float designPx, ImU32 color, float wrap) {
            if (text.empty()) return;
            dl->AddText(bodyFont(), cells::fontPx(designPx), ImVec2(theme::snap(pos.x), theme::snap(pos.y)), color, text.c_str(),
                        text.c_str() + text.size(), std::max(1.0f, wrap));
        }

        // The plots are pictures, not instruments: no frame, no padding, no
        // axes, no interaction: flat cells.
        struct PlotLook {
            PlotLook() {
                ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
                ImPlot::PushStyleVar(ImPlotStyleVar_PlotMinSize, ImVec2(1, 1));
                ImPlot::PushStyleVar(ImPlotStyleVar_PlotBorderSize, 0.0f);
                ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0, 0, 0, 0));
                ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(0, 0, 0, 0));
                ImPlot::PushStyleColor(ImPlotCol_PlotBorder, ImVec4(0, 0, 0, 0));
            }
            ~PlotLook() {
                ImPlot::PopStyleColor(3);
                ImPlot::PopStyleVar(3);
            }
            PlotLook(const PlotLook&) = delete;
            PlotLook& operator=(const PlotLook&) = delete;
        };

        constexpr ImPlotFlags kPlotFlags = ImPlotFlags_CanvasOnly | ImPlotFlags_NoInputs | ImPlotFlags_NoFrame;
        constexpr ImPlotAxisFlags kAxisFlags = ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_NoMenus;

    } // namespace

    // --- image rendering -------------------------------------------------------

    GrayImage renderDiagnosticImage(const DiagnosticImage& image) {
        GrayImage out;
        if (image.rows <= 0 || image.cols <= 0 || image.values.size() < static_cast<std::size_t>(image.rows * image.cols)) return out;
        if (image.rows > std::numeric_limits<int>::max() || image.cols > std::numeric_limits<int>::max()) return out;
        const Index n = image.rows * image.cols;
        // robust window from a bounded sample
        std::vector<float> sample;
        const Index stride = std::max<Index>(1, n / 65536);
        sample.reserve(static_cast<std::size_t>(n / stride + 1));
        for (Index i = 0; i < n; i += stride) {
            const float v = image.values[static_cast<std::size_t>(i)];
            if (std::isfinite(v)) sample.push_back(v);
        }
        float lo = 0.0f, hi = 1.0f;
        if (!sample.empty()) {
            auto rank = [&](double frac) {
                return static_cast<std::ptrdiff_t>(std::llround(frac * static_cast<double>(sample.size() - 1)));
            };
            const std::ptrdiff_t kLo = rank(0.005), kHi = rank(0.998);
            std::nth_element(sample.begin(), sample.begin() + kLo, sample.end());
            lo = sample[static_cast<std::size_t>(kLo)];
            std::nth_element(sample.begin() + kLo, sample.begin() + kHi, sample.end());
            hi = sample[static_cast<std::size_t>(kHi)];
            if (!(hi > lo)) {
                const auto [mn, mx] = std::minmax_element(sample.begin(), sample.end());
                lo = *mn;
                hi = *mx;
            }
            if (!(hi > lo)) hi = lo + 1.0f;
        }
        out.width = static_cast<int>(image.cols);
        out.height = static_cast<int>(image.rows);
        out.pixels.resize(static_cast<std::size_t>(n));
        const float scale = 255.0f / (hi - lo);
        for (Index y = 0; y < image.rows; ++y) {
            std::uint8_t* row = out.pixels.data() + y * image.cols;
            const float* src = image.values.data() + y * image.cols;
            for (Index x = 0; x < image.cols; ++x) {
                const float t = std::min(255.0f, std::max(0.0f, (src[x] - lo) * scale));
                row[x] = static_cast<std::uint8_t>(std::isfinite(t) ? t + 0.5f : 0.0f);
            }
        }
        return out;
    }

    namespace cells {

        Rect Rect::inset(float left, float top, float right, float bottom) const {
            Rect r;
            r.min = ImVec2(min.x + px(left), min.y + px(top));
            r.max = ImVec2(std::max(r.min.x, max.x - px(right)), std::max(r.min.y, max.y - px(bottom)));
            return r;
        }

        float fontPx(float designPx) {
            const ImGuiStyle& st = ImGui::GetStyle();
            return designPx * st.FontScaleMain * st.FontScaleDpi;
        }

        std::string elideIn(ImFont* face, float designPx, const std::string& text, float width) {
            if (!face) face = ImGui::GetFont();
            const float size = fontPx(designPx);
            auto measure = [&](const char* b, const char* e) { return face->CalcTextSizeA(size, FLT_MAX, 0.0f, b, e).x; };
            if (measure(text.c_str(), text.c_str() + text.size()) <= width) return text;
            static const std::string dots = "…";
            const float room = width - measure(dots.c_str(), dots.c_str() + dots.size());
            if (room <= 0.0f) return dots;
            std::size_t fit = 0;
            for (std::size_t i = 0; i < text.size();) {
                nextCodepoint(text, i);
                if (measure(text.c_str(), text.c_str() + i) > room) break;
                fit = i;
            }
            std::string out = text.substr(0, fit);
            while (!out.empty() && out.back() == ' ') out.pop_back();
            return out + dots;
        }

        float captionRowHeight() {
            // 6 px above, 4 px below the 10 px caption
            return theme::snap(px(6) + std::ceil(fontPx(theme::kCaptionPx) * 1.3f) + px(4));
        }

        Rect beginBox(const char* id, ImVec2 min, ImVec2 max) {
            const ImVec2 size(std::max(1.0f, max.x - min.x), std::max(1.0f, max.y - min.y));
            ImGui::SetCursorScreenPos(min);
            ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kBg);
            ImGui::BeginChild(id, size, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            ImGui::PopStyleColor();
            Rect r;
            r.min = min;
            r.max = ImVec2(min.x + size.x, min.y + size.y);
            return r;
        }

        Rect beginCell(const char* id, ImVec2 min, ImVec2 max, const std::string& title, const std::string& meta) {
            Rect r = beginBox(id, min, max);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float rowH = captionRowHeight();
            const float size = fontPx(theme::kCaptionPx);
            const float left = r.min.x + px(12), right = r.max.x - px(12);
            const float textY = theme::snap(r.min.y + px(6) + (rowH - px(10) - size) * 0.5f);
            // the meta keeps its place; the title gives way
            const std::string metaText = captionCase(meta);
            ImFont* plain = bodyFont();
            float metaW = 0.0f;
            if (!metaText.empty()) {
                metaW = std::ceil(plain->CalcTextSizeA(size, FLT_MAX, 0.0f, metaText.c_str(), metaText.c_str() + metaText.size()).x);
                metaW = std::min(metaW, std::max(0.0f, right - left));
                const std::string shown = elideIn(plain, theme::kCaptionPx, metaText, metaW);
                dl->AddText(plain, size, ImVec2(theme::snap(right - metaW), textY), theme::kNeutral600, shown.c_str(),
                            shown.c_str() + shown.size());
            }
            const float room = right - left - (metaW > 0.0f ? metaW + px(8) : 0.0f);
            if (room > px(8)) {
                const std::string whole = captionCase(title);
                const std::string shown = elideIn(theme::captionFont(), theme::kCaptionPx, whole, room);
                dl->AddText(theme::captionFont(), size, ImVec2(theme::snap(left), textY), theme::kNeutral600, shown.c_str(),
                            shown.c_str() + shown.size());
                if (shown != whole) {
                    // a caption that was cut says the rest when pointed at
                    ImGui::SetCursorScreenPos(ImVec2(left, r.min.y));
                    ImGui::Dummy(ImVec2(room, rowH));
                    widgets::tooltip(whole);
                }
            }
            r.min.y = std::min(r.max.y, r.min.y + rowH);
            ImGui::SetCursorScreenPos(r.min);
            return r;
        }

        void endCell() {
            // The content positions itself with SetCursorScreenPos; an empty
            // item at the origin tells Dear ImGui the cell grows no further.
            ImGui::SetCursorPos(ImVec2(0, 0));
            ImGui::Dummy(ImVec2(0, 0));
            ImGui::EndChild();
        }

        // --- image -------------------------------------------------------------

        void image(const Rect& r, Texture* texture, const std::vector<DiagnosticMark>& marks, const std::string& placeholder) {
            if (r.width() < 1.0f || r.height() < 1.0f) return;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(r.min, r.max, theme::kViewerGround);
            if (!texture || !texture->valid() || texture->width() <= 0 || texture->height() <= 0) {
                if (!placeholder.empty())
                    drawWrappedCentered(dl, r.inset(10, 8, 10, 8), placeholder, theme::kSmallPx, theme::withAlpha(theme::kViewerText, 0.55f));
                return;
            }
            const float sx = r.width() / static_cast<float>(texture->width());
            const float sy = r.height() / static_cast<float>(texture->height());
            const float s = std::min(sx, sy);
            const float w = static_cast<float>(texture->width()) * s, h = static_cast<float>(texture->height()) * s;
            const ImVec2 min(r.min.x + (r.width() - w) * 0.5f, r.min.y + (r.height() - h) * 0.5f);
            const ImVec2 max(min.x + w, min.y + h);
            // pixels stay pixels when enlarged, and are averaged when reduced
            texture->setSmooth(s < 1.0f);
            texture->draw(dl, min, max);
            if (marks.empty()) return;
            dl->PushClipRect(r.min, r.max, true);
            const float pen = std::max(1.0f, px(1.2f));
            for (const DiagnosticMark& m : marks) {
                const ImU32 c = m.accent ? theme::kAccent : theme::kViewerText;
                const ImVec2 at(min.x + static_cast<float>(m.x) * s, min.y + static_cast<float>(m.y) * s);
                const float radius = std::max(px(2), static_cast<float>(m.radius) * s);
                switch (m.kind) {
                    case DiagnosticMark::Kind::Cross:
                        dl->AddLine(ImVec2(at.x - px(5), at.y), ImVec2(at.x + px(5), at.y), c, pen);
                        dl->AddLine(ImVec2(at.x, at.y - px(5)), ImVec2(at.x, at.y + px(5)), c, pen);
                        break;
                    case DiagnosticMark::Kind::Circle:
                        dl->AddCircleFilled(at, radius, theme::withAlpha(c, 0.35f), 0);
                        dl->AddCircle(at, radius, c, 0, pen);
                        break;
                    case DiagnosticMark::Kind::Ring:
                        dl->AddCircle(at, radius, c, 0, pen);
                        break;
                }
                if (!m.text.empty()) {
                    // the label's baseline sits at (7, -4) from the mark
                    const float size = fontPx(theme::kCaptionPx);
                    dl->AddText(bodyFont(), size, ImVec2(theme::snap(at.x + px(7)), theme::snap(at.y - px(4) - size * 0.85f)), c,
                                m.text.c_str(), m.text.c_str() + m.text.size());
                }
            }
            dl->PopClipRect();
        }

        // --- curve ---------------------------------------------------------------

        void curve(const char* id, const Rect& r, const DiagnosticCurve* c) {
            if (r.width() < 1.0f || r.height() < 1.0f) return;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float labelH = 16;
            const Rect plot = r.inset(14, 4, 14, labelH + 6);
            if (plot.width() < 2.0f || plot.height() < 2.0f) return;
            // the baseline, the pen centred on the plot's bottom edge
            const float pen = theme::crispPen(2);
            dl->AddRectFilled(ImVec2(plot.min.x, theme::snap(plot.max.y) - pen * 0.5f), ImVec2(plot.max.x, theme::snap(plot.max.y) + pen * 0.5f),
                              theme::kText);
            if (!c || c->y.empty()) {
                drawWrappedCentered(dl, plot, "Run the step to record the curve", theme::kSmallPx, theme::kNeutral600);
                return;
            }
            const std::size_t n = c->y.size();
            const bool ownX = c->x.size() == n;
            std::vector<double> xs(n), ys(n);
            for (std::size_t i = 0; i < n; ++i) {
                xs[i] = ownX ? c->x[i] : static_cast<double>(i);
                // a log axis has no place for zero: floored at 1e-12
                ys[i] = c->logY ? std::max(c->y[i], 1e-12) : c->y[i];
            }
            double xmin = 0.0, xmax = static_cast<double>(n - 1);
            if (ownX && n > 1) {
                xmin = *std::min_element(xs.begin(), xs.end());
                xmax = *std::max_element(xs.begin(), xs.end());
            }
            if (xmax <= xmin) xmax = xmin + 1.0;
            auto yval = [&](double y) { return c->logY ? std::log10(std::max(y, 1e-12)) : y; };
            double ymin = std::numeric_limits<double>::infinity(), ymax = -ymin;
            for (double y : c->y) {
                const double v = yval(y);
                if (!std::isfinite(v)) continue;
                ymin = std::min(ymin, v);
                ymax = std::max(ymax, v);
            }
            if (!std::isfinite(ymin)) {
                ymin = 0.0;
                ymax = 1.0;
            }
            if (!c->logY) ymin = std::min(ymin, 0.0);
            if (ymax <= ymin) ymax = ymin + 1.0;
            // the highest point sits 2 px under the top edge, where the pen is whole
            const double h = static_cast<double>(plot.height());
            const double top = ymin + (ymax - ymin) * h / std::max(1.0, h - static_cast<double>(px(2)));

            ImGui::SetCursorScreenPos(plot.min);
            {
                const PlotLook look;
                if (ImPlot::BeginPlot(id, ImVec2(plot.width(), plot.height()), kPlotFlags)) {
                    ImPlot::SetupAxes(nullptr, nullptr, kAxisFlags, kAxisFlags);
                    if (c->logY) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
                    ImPlot::SetupAxisLimits(ImAxis_X1, xmin, xmax, ImPlotCond_Always);
                    if (c->logY) ImPlot::SetupAxisLimits(ImAxis_Y1, std::pow(10.0, ymin), std::pow(10.0, top), ImPlotCond_Always);
                    else ImPlot::SetupAxisLimits(ImAxis_Y1, ymin, top, ImPlotCond_Always);
                    ImPlotSpec spec;
                    spec.LineColor = theme::vec(theme::kAccent);
                    spec.LineWeight = std::max(1.0f, px(2));
                    spec.Flags = ImPlotItemFlags_NoLegend;
                    ImPlot::PlotLine("##y", xs.data(), ys.data(), static_cast<int>(n), spec);
                    if (c->stopX) {
                        // ImPlot has no dashed pen: the stop line is drawn in the plot's own list
                        const float x = theme::snap(ImPlot::PlotToPixels(*c->stopX, c->logY ? std::pow(10.0, ymin) : ymin).x);
                        ImDrawList* pdl = ImPlot::GetPlotDrawList();
                        ImPlot::PushPlotClipRect();
                        const float dash = std::max(1.0f, px(3)), w = theme::crispPen(1);
                        for (float y = plot.min.y; y < plot.max.y; y += 2 * dash)
                            pdl->AddRectFilled(ImVec2(x, y), ImVec2(x + w, std::min(y + dash, plot.max.y)), theme::kText);
                        ImPlot::PopPlotClipRect();
                    }
                    ImPlot::EndPlot();
                }
            }
            const ImVec2 lmin(plot.min.x, plot.max.y + px(4)), lmax(plot.max.x, plot.max.y + px(4) + px(labelH));
            if (!c->leftLabel.empty()) widgets::drawTextIn(dl, lmin, lmax, c->leftLabel, theme::kSmallPx, theme::kNeutral600, Weight::Regular, 0.0f, 0.5f);
            if (!c->midLabel.empty()) widgets::drawTextIn(dl, lmin, lmax, c->midLabel, theme::kSmallPx, theme::kNeutral600, Weight::Regular, 0.5f, 0.5f);
            if (!c->rightLabel.empty()) widgets::drawTextIn(dl, lmin, lmax, c->rightLabel, theme::kSmallPx, theme::kNeutral600, Weight::Regular, 1.0f, 0.5f);
        }

        // --- histogram -----------------------------------------------------------

        void histogram(const char* id, const Rect& r, const DiagnosticHistogram* hist) {
            if (r.width() < 1.0f || r.height() < 1.0f) return;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const Rect inner = r.inset(14, 4, 14, 8);
            const float headerH = px(18);
            if (!hist) {
                drawWrappedCentered(dl, inner, "No intensity data yet", theme::kSmallPx, theme::kNeutral600);
                return;
            }
            const DiagnosticHistogram& h = *hist;
            dl->PushClipRect(r.min, r.max, true);
            // header: chip, label, window
            const ImVec2 chip(theme::snap(inner.min.x), theme::snap(inner.min.y + px(4)));
            dl->AddRectFilled(chip, ImVec2(chip.x + theme::snap(px(10)), chip.y + theme::snap(px(10))), theme::fromFloat(h.color));
            const float labelX = inner.min.x + px(18);
            widgets::drawTextIn(dl, ImVec2(labelX, inner.min.y), ImVec2(inner.max.x, inner.min.y + headerH), h.channel, theme::kSmallPx,
                                theme::kText, Weight::ExtraBold, 0.0f, 0.5f);
            const float labelW = theme::textSize(h.channel, theme::kSmallPx, Weight::ExtraBold).x;
            const std::string window = formatNumber(h.lo) + " – " + formatNumber(h.hi) + " · γ " + format("%.3g", h.gamma);
            const float windowX = labelX + labelW + px(8);
            widgets::drawTextIn(dl, ImVec2(windowX, inner.min.y), ImVec2(std::max(windowX, inner.max.x), inner.min.y + headerH), window,
                                theme::kSmallPx, theme::kNeutral600, Weight::Regular, 0.0f, 0.5f);
            dl->PopClipRect();
            // bars
            Rect bars;
            bars.min = ImVec2(inner.min.x, inner.min.y + headerH + px(6));
            bars.max = inner.max;
            if (bars.height() < 3.0f || bars.width() < 3.0f) return;
            const float pen = theme::crispPen(2);
            dl->AddRectFilled(ImVec2(bars.min.x, theme::snap(bars.max.y) - pen * 0.5f), ImVec2(bars.max.x, theme::snap(bars.max.y) + pen * 0.5f),
                              theme::kDivider);
            if (h.bins.empty()) return;
            const double maxBin = std::max(1e-12, *std::max_element(h.bins.begin(), h.bins.end()));
            const int n = static_cast<int>(h.bins.size());
            // the bars stand on the rule, not in it
            Rect plot = bars;
            plot.max.y = theme::snap(bars.max.y) - pen * 0.5f;
            const double height = static_cast<double>(plot.height());
            const double width = static_cast<double>(plot.width());
            if (height < 2.0) return;
            // the tallest bar ends 1 px under the top; every bin shows at least 1 px
            const double top = maxBin * height / std::max(1.0, height - 1.0);
            const double floor1 = top / height;
            const double binW = (h.binHi - h.binLo) / n;
            std::vector<double> xs(static_cast<std::size_t>(n)), ys(static_cast<std::size_t>(n));
            std::vector<ImU32> colours(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                const std::size_t k = static_cast<std::size_t>(i);
                const double centre = h.binLo + (i + 0.5) * binW;
                const bool tail = centre < h.lo || centre > h.hi;
                const double v = std::isfinite(h.bins[k]) ? h.bins[k] : 0.0;
                xs[k] = static_cast<double>(i);
                ys[k] = std::max(v, floor1);
                colours[k] = tail ? theme::kNeutral400 : theme::kText;
            }
            // A 2 px gap between the bars, the first and the last flush with
            // the edges: with bars one unit apart and `bar` units wide the
            // axis spans n - 1 + bar units over `width` pixels.
            const double gap = static_cast<double>(px(2));
            const double bar = std::clamp((width - gap * (n - 1)) / (width + gap), 0.1, 1.0);
            ImGui::SetCursorScreenPos(plot.min);
            {
                const PlotLook look;
                if (ImPlot::BeginPlot(id, ImVec2(plot.width(), plot.height()), kPlotFlags)) {
                    ImPlot::SetupAxes(nullptr, nullptr, kAxisFlags, kAxisFlags);
                    ImPlot::SetupAxisLimits(ImAxis_X1, -bar * 0.5, static_cast<double>(n - 1) + bar * 0.5, ImPlotCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, top, ImPlotCond_Always);
                    ImPlotSpec spec;
                    spec.FillColor = theme::vec(theme::kText);
                    spec.FillColors = colours.data();
                    spec.FillAlpha = 1.0f;
                    spec.LineColor = ImVec4(0, 0, 0, 0);
                    spec.LineWeight = 0.0f;
                    spec.Flags = ImPlotItemFlags_NoLegend;
                    ImPlot::PlotBars("##bins", xs.data(), ys.data(), n, bar, spec);
                    ImPlot::EndPlot();
                }
            }
        }

        // --- facts -----------------------------------------------------------------

        void facts(const char* id, const Rect& r, const std::vector<DiagnosticFact>& list, const std::string& lead,
                   const std::string& trailer) {
            if (r.width() < 1.0f || r.height() < 1.0f) return;
            ImGui::SetCursorScreenPos(r.min);
            // more facts than the cell is high: the cell scrolls
            if (ImGui::BeginChild(id, ImVec2(r.width(), r.height()), ImGuiChildFlags_None, ImGuiWindowFlags_None)) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                const ImVec2 origin = ImGui::GetCursorScreenPos();
                const float left = origin.x + px(14), right = origin.x + r.width() - px(14);
                const float wrap = std::max(px(20), right - left);
                float y = origin.y + px(4);
                if (!lead.empty()) {
                    drawWrapped(dl, ImVec2(left, y), lead, theme::kBodyPx, theme::kText, wrap);
                    y += wrappedHeight(lead, theme::kBodyPx, wrap) + px(8);
                }
                const float rowH = theme::snap(px(24));
                for (const DiagnosticFact& f : list) {
                    const ImVec2 rmin(left, y), rmax(right, y + rowH);
                    // the value keeps its place; a long key gives way to it
                    const float valueW = theme::textSize(f.value, 12).x;
                    const float keyRoom = std::max(px(20), right - left - valueW - px(8));
                    widgets::drawTextIn(dl, rmin, rmax, widgets::elideText(f.key, keyRoom, 12), 12, theme::kText, Weight::Regular, 0.0f, 0.5f);
                    widgets::drawTextIn(dl, rmin, rmax, widgets::elideText(f.value, std::max(px(20), right - left), 12), 12, theme::kText,
                                        Weight::Regular, 1.0f, 0.5f);
                    dl->AddRectFilled(ImVec2(left, y + rowH - theme::crispPen(1)), ImVec2(right, y + rowH), theme::kDivider);
                    y += rowH;
                }
                if (!trailer.empty()) {
                    drawWrapped(dl, ImVec2(left, y + px(6)), trailer, theme::kSmallPx, theme::kNeutral600, wrap);
                    y += px(6) + wrappedHeight(trailer, theme::kSmallPx, wrap);
                }
                ImGui::SetCursorScreenPos(origin);
                ImGui::Dummy(ImVec2(1.0f, y - origin.y + px(4)));
            }
            ImGui::EndChild();
        }

        // --- tile map ----------------------------------------------------------------

        void tileMap(const Rect& r, const AlignmentInfo& info) {
            if (r.width() < 1.0f || r.height() < 1.0f) return;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const Index rows = std::max<Index>(info.gridRows, 1), cols = std::max<Index>(info.gridCols, 1);
            const Rect area = r.inset(12, 8, 12, 8);
            const float gap = px(4);
            const float tw = (area.width() - gap * static_cast<float>(cols - 1)) / static_cast<float>(cols);
            const float th = (area.height() - gap * static_cast<float>(rows - 1)) / static_cast<float>(rows);
            if (tw < 1.0f || th < 1.0f) return;
            dl->PushClipRect(r.min, r.max, true);
            for (Index row = 0; row < rows; ++row)
                for (Index col = 0; col < cols; ++col) {
                    const Index i = row * cols + col;
                    const ImVec2 min(area.min.x + static_cast<float>(col) * (tw + gap), area.min.y + static_cast<float>(row) * (th + gap));
                    const ImVec2 max(min.x + tw, min.y + th);
                    const bool hi = info.highlightedTile >= 0 && i == static_cast<Index>(info.highlightedTile);
                    dl->AddRectFilled(ImVec2(theme::snap(min.x), theme::snap(min.y)), ImVec2(theme::snap(max.x), theme::snap(max.y)),
                                      hi ? theme::kAccent : theme::kBg);
                    widgets::crispRect(dl, min, max, theme::kText, theme::kBorder);
                    const std::string name = i < static_cast<Index>(info.tileNames.size()) ? info.tileNames[static_cast<std::size_t>(i)]
                                                                                           : format("t%lld", static_cast<long long>(i + 1));
                    const std::string shown = widgets::elideText(name, std::max(1.0f, tw - px(6)), theme::kSmallPx);
                    widgets::drawTextIn(dl, min, max, shown, theme::kSmallPx, hi ? theme::kBg : theme::kText);
                }
            dl->PopClipRect();
        }

        // --- tables --------------------------------------------------------------------

        float tableRowHeight() { return theme::snap(px(22)); }

        float tableHeaderHeight() {
            // 10 px text in the same 4 px of padding the body rows have
            return theme::snap(std::ceil(fontPx(theme::kCaptionPx)) + 2.0f * theme::snap(px(5)));
        }

        void pushTableStyle() {
            ImGui::PushFont(bodyFont(), theme::kSmallPx);
            // rows of 22 px whatever the face's line height is
            const float pad = std::max(1.0f, std::floor((tableRowHeight() - ImGui::GetTextLineHeight()) * 0.5f));
            ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(px(6), pad));
            ImGui::PushStyleColor(ImGuiCol_Header, theme::kSurface);
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, theme::kNeutral200);
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, theme::kNeutral300);
            ImGui::PushStyleColor(ImGuiCol_TableBorderLight, theme::kDivider);
            ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, theme::kDivider);
            ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, theme::kBg);
        }

        void popTableStyle() {
            ImGui::PopStyleColor(6);
            ImGui::PopStyleVar();
            ImGui::PopFont();
        }

        void tableHeaders(const char* const* tips) {
            const int n = ImGui::TableGetColumnCount();
            const float rowH = tableHeaderHeight();
            const theme::FontScope f(theme::kCaptionPx);
            const float pad = std::max(1.0f, std::floor((rowH - ImGui::GetTextLineHeight()) * 0.5f));
            ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(px(6), pad));
            ImGui::PushStyleColor(ImGuiCol_Text, theme::kNeutral600);
            ImGui::TableNextRow(ImGuiTableRowFlags_Headers, rowH);
            for (int c = 0; c < n; ++c) {
                if (!ImGui::TableSetColumnIndex(c)) continue;
                ImGui::PushID(c);
                const char* name = ImGui::TableGetColumnName(c);
                ImGui::TableHeader(name ? name : "");
                if (tips && tips[c]) {
                    ImGui::PushStyleColor(ImGuiCol_Text, theme::kText);
                    widgets::tooltip(tips[c]);
                    ImGui::PopStyleColor();
                }
                ImGui::PopID();
            }
            ImGui::PopStyleColor();
            ImGui::PopStyleVar();
        }

        void tableHeaderRule(ImVec2 tableMin, float tableWidth, float headerHeight) {
            const float y = theme::snap(tableMin.y + headerHeight);
            ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(tableMin.x, y - theme::crispPen(2)), ImVec2(tableMin.x + tableWidth, y), theme::kDivider);
        }

        void table(const char* id, const Rect& r, const DiagnosticTable& t) {
            const int columns = static_cast<int>(t.header.size());
            if (columns <= 0 || r.width() < 1.0f || r.height() < 1.0f) return;
            ImGui::SetCursorScreenPos(r.min);
            pushTableStyle();
            const float headerH = tableHeaderHeight();
            const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_PadOuterX | ImGuiTableFlags_SizingFixedFit |
                                          ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_NoHostExtendX;
            bool drawn = false;
            if (ImGui::BeginTable(id, columns, flags, ImVec2(r.width(), r.height()))) {
                drawn = true;
                ImGui::TableSetupScrollFreeze(0, 1);
                for (int c = 0; c < columns; ++c) {
                    const std::string name = captionCase(t.header[static_cast<std::size_t>(c)]) + "##" + std::to_string(c);
                    ImGui::TableSetupColumn(name.c_str(), (c + 1 == columns ? ImGuiTableColumnFlags_WidthStretch : ImGuiTableColumnFlags_WidthFixed) |
                                                              ImGuiTableColumnFlags_NoSort);
                }
                tableHeaders();
                for (std::size_t row = 0; row < t.rows.size(); ++row) {
                    ImGui::TableNextRow();
                    const std::vector<std::string>& cellsOf = t.rows[row];
                    for (int c = 0; c < columns && c < static_cast<int>(cellsOf.size()); ++c) {
                        if (!ImGui::TableSetColumnIndex(c)) continue;
                        const std::string& text = cellsOf[static_cast<std::size_t>(c)];
                        const bool accent = std::find(t.accentCells.begin(), t.accentCells.end(), std::make_pair(static_cast<int>(row), c)) !=
                                            t.accentCells.end();
                        if (accent) widgets::text(text, theme::kSmallPx, theme::kAccentText, Weight::ExtraBold);
                        else widgets::text(text, theme::kSmallPx, theme::kText);
                    }
                }
                ImGui::EndTable();
            }
            popTableStyle();
            if (drawn) tableHeaderRule(r.min, r.width(), headerH);
        }

        void tooltipAlways(const std::string& text) {
            if (text.empty() || !ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) return;
            ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
            ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(theme::kBorder));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(8, 4));
            // a tool tip raised inside BeginDisabled would be drawn at 45 %
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 1.0f);
            if (ImGui::BeginTooltip()) {
                {   // the font is popped inside the tooltip it was pushed in
                    const theme::FontScope f(12);
                    ImGui::PushStyleColor(ImGuiCol_Text, theme::kText);
                    ImGui::PushTextWrapPos(px(360));
                    ImGui::TextUnformatted(text.c_str(), text.c_str() + text.size());
                    ImGui::PopTextWrapPos();
                    ImGui::PopStyleColor();
                }
                ImGui::EndTooltip();
            }
            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor();
        }

    } // namespace cells

    // --- DiagnosticsBody -------------------------------------------------------------

    void DiagnosticsBody::setDiagnostics(Diagnostics d, DiagnosticsKind kind, Context ctx) {
        d_ = std::move(d);
        kind_ = kind;
        ctx_ = std::move(ctx);
        // The textures keep their storage (an image of the same size is
        // uploaded into it); what they show is rendered again on first use.
        if (textures_.size() != d_.images.size()) {
            textures_.clear();
            textures_.resize(d_.images.size());
        }
        rendered_.assign(d_.images.size(), 0);
    }

    Texture* DiagnosticsBody::texture(std::size_t index) {
        if (index >= d_.images.size() || index >= textures_.size()) return nullptr;
        if (!rendered_[index]) {
            rendered_[index] = 1;
            const GrayImage g = renderDiagnosticImage(d_.images[index]);
            if (g.empty()) textures_[index].reset();
            else textures_[index].uploadGray(g.pixels.data(), g.width, g.height, false);
        }
        return textures_[index].valid() ? &textures_[index] : nullptr;
    }

    std::vector<std::string> DiagnosticsBody::tabNames(const Diagnostics& d, DiagnosticsKind kind) {
        std::vector<std::string> names;
        for (const DiagnosticTab& t : d.tabs) names.push_back(t.name);
        if (!names.empty()) return names;
        switch (kind) {
            case DiagnosticsKind::Sim: return {"Raw spectrum", "Separated bands", "Wiener-filtered bands", "Result spectrum"};
            case DiagnosticsKind::Deconvolve: return {"Convergence"};
            case DiagnosticsKind::Contrast: return {"Histograms"};
            case DiagnosticsKind::Segment: return {"Cleanup"};
            case DiagnosticsKind::Volume: return {"Rendering"};
            case DiagnosticsKind::Alignment: return {"Alignment"};
            case DiagnosticsKind::Generic: break;
        }
        return {"Preview"};
    }

    void DiagnosticsBody::imageCell(std::vector<Cell>& out, std::size_t index, const std::string& fallbackTitle,
                                    const std::string& fallbackMeta, const std::string& placeholder, float fixedWidth) {
        Cell cell;
        const bool has = index < d_.images.size();
        cell.title = has ? d_.images[index].title : fallbackTitle;
        cell.meta = has ? d_.images[index].meta : fallbackMeta;
        cell.stretch = fixedWidth > 0.0f ? 0 : 1;
        cell.fixedWidth = fixedWidth;
        cell.content = [this, index, placeholder](const cells::Rect& r) {
            static const std::vector<DiagnosticMark> none;
            const bool present = index < d_.images.size();
            // An image that is there but empty (a plane of no pixels) is a
            // black cell; the placeholder is for the missing one.
            cells::image(r, present ? texture(index) : nullptr, present ? d_.images[index].marks : none, present ? std::string() : placeholder);
        };
        out.push_back(std::move(cell));
    }

    void DiagnosticsBody::draw(int tab) {
        const Diagnostics& d = d_;
        // images of the active tab (all images without tabs)
        std::vector<std::size_t> tabImages;
        if (!d.tabs.empty()) {
            const std::size_t t = static_cast<std::size_t>(std::clamp(tab, 0, static_cast<int>(d.tabs.size()) - 1));
            for (int i : d.tabs[t].images)
                if (i >= 0 && static_cast<std::size_t>(i) < d.images.size()) tabImages.push_back(static_cast<std::size_t>(i));
        } else {
            for (std::size_t i = 0; i < d.images.size(); ++i) tabImages.push_back(i);
        }
        const std::size_t missing = d.images.size();   // an index no image has

        std::vector<Cell> cellsOf;
        auto add = [&](const std::string& title, const std::string& meta, std::function<void(const cells::Rect&)> content, int stretch,
                       float fixedWidth = 0.0f) {
            Cell cell;
            cell.title = title;
            cell.meta = meta;
            cell.stretch = stretch;
            cell.fixedWidth = fixedWidth;
            cell.content = std::move(content);
            cellsOf.push_back(std::move(cell));
        };
        auto curveCell = [&](const std::string& title, std::size_t index) {
            add(title, {}, [this, index](const cells::Rect& r) { cells::curve("##curve", r, index < d_.curves.size() ? &d_.curves[index] : nullptr); }, 1);
        };
        auto histogramCell = [&](const std::string& title, std::size_t index) {
            add(title, {}, [this, index](const cells::Rect& r) { cells::histogram("##histogram", r, index < d_.histograms.size() ? &d_.histograms[index] : nullptr); }, 1);
        };

        switch (kind_) {
            case DiagnosticsKind::Sim: {
                static const char* kPlaceholders[4][3] = {
                    {"Raw FFT · phase 1", "Raw FFT · phase 2", "Raw FFT · phase 3"},
                    {"Order 1 · angle 1", "Order 1 · angle 2", "Order 1 · angle 3"},
                    {"Filtered order 1 · angle 1", "Filtered order 1 · angle 2", "Filtered order 1 · angle 3"},
                    {"Widefield", "SIM result", "Difference"},
                };
                const int t = std::clamp(tab, 0, 3);
                for (std::size_t i = 0; i < 3; ++i)
                    imageCell(cellsOf, i < tabImages.size() ? tabImages[i] : missing, kPlaceholders[t][i], {}, "Run the step to see the spectrum");
                add(d.table ? d.table->caption : std::string("Estimated parameters"), {}, [this](const cells::Rect& r) {
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    const float wrap = std::max(px(20), r.width() - px(24));
                    // the footer keeps its height at the bottom, the table takes the rest
                    float footerH = 0.0f;
                    if (!d_.footer.empty()) {
                        footerH = wrappedHeight(d_.footer, theme::kSmallPx, wrap) + px(8);
                        drawWrapped(dl, ImVec2(r.min.x + px(12), r.max.y - footerH), d_.footer, theme::kSmallPx, theme::kNeutral600, wrap);
                        footerH += px(6);
                    }
                    cells::Rect top = r;
                    top.max.y = std::max(r.min.y, r.max.y - footerH);
                    if (d_.table) cells::table("##table", top, *d_.table);
                    else
                        drawWrapped(dl, ImVec2(r.min.x + px(12), r.min.y),
                                    "Run the step to fit the pattern vectors and modulation depths.", theme::kSmallPx, theme::kText, wrap); }, 0, 300);
                break;
            }
            case DiagnosticsKind::Deconvolve: {
                std::string title = "Convergence · relative change per iteration";
                if (!d.curves.empty() && !d.curves.front().title.empty()) title = d.curves.front().title;
                curveCell(title, 0);
                imageCell(cellsOf, 0, "PSF · XZ", {}, "PSF", 260);
                imageCell(cellsOf, 1, "Residual", {}, "Residual after the run", 260);
                break;
            }
            case DiagnosticsKind::Contrast: {
                if (d.histograms.empty()) histogramCell("Histograms", 0);
                else
                    for (std::size_t i = 0; i < d.histograms.size(); ++i) histogramCell(d.histograms[i].channel, i);
                break;
            }
            case DiagnosticsKind::Alignment: {
                imageCell(cellsOf, 0, "Checkerboard · fixed ⇄ moving", {}, "Run the step to compare fixed and moving");
                add("Tile layout", {}, [this](const cells::Rect& r) {
                    if (d_.alignment) {
                        cells::tileMap(r, *d_.alignment);
                    } else {
                        AlignmentInfo none;
                        none.gridRows = 1;
                        none.gridCols = 1;
                        none.tileNames = {"–"};
                        cells::tileMap(r, none);
                    } }, 1);
                add("Pairwise shifts", {}, [this](const cells::Rect& r) { cells::facts("##facts", r, d_.alignment ? d_.alignment->shiftStats : d_.facts, {},
                                                                                       d_.alignment ? std::string() : std::string("Pairwise shifts appear after the run.")); }, 0, 300);
                break;
            }
            case DiagnosticsKind::Volume: {
                curveCell("Transfer function", 0);
                add("Reconstruction", {}, [this](const cells::Rect& r) { cells::facts("##facts", r, d_.facts); }, 1);
                imageCell(cellsOf, 0, "Isosurface preview", {}, "Preview after the run");
                break;
            }
            case DiagnosticsKind::Segment:
            case DiagnosticsKind::Generic: {
                const std::size_t in = tabImages.size() > 0 ? tabImages[0] : missing;
                const std::size_t out = tabImages.size() > 1 ? tabImages[1] : missing;
                imageCell(cellsOf, in, "Input", ctx_.inputShape, "Input preview");
                imageCell(cellsOf, out, "Output · live", ctx_.outputShape, "Output preview after the run");
                add("Step summary", {}, [this](const cells::Rect& r) { cells::facts("##facts", r, d_.facts, d_.summary.empty() ? ctx_.stepSummary : d_.summary, ctx_.estimate); }, 1);
                break;
            }
        }
        // extra curves / histograms of a generic diagnostics land in more cells
        if (kind_ == DiagnosticsKind::Generic) {
            for (std::size_t i = 0; i < d.curves.size(); ++i) curveCell(d.curves[i].title, i);
            for (std::size_t i = 0; i < d.histograms.size(); ++i) histogramCell(d.histograms[i].channel, i);
        }

        // --- the grid: one row, 2 px of divider between the cells ---------------
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        if (avail.x < 2.0f || avail.y < 2.0f || cellsOf.empty()) return;
        ImGui::GetWindowDrawList()->AddRectFilled(origin, ImVec2(origin.x + avail.x, origin.y + avail.y), theme::kDivider);
        const float gap = theme::crispPen(2);
        const float room = avail.x - gap * static_cast<float>(cellsOf.size() - 1);
        float fixed = 0.0f;
        int stretch = 0;
        for (const Cell& c : cellsOf) {
            if (c.fixedWidth > 0.0f) fixed += px(c.fixedWidth);
            else stretch += std::max(1, c.stretch);
        }
        // A dock too narrow for the fixed cells squeezes them rather than
        // pushing the stretching ones out: every cell stays on screen.
        const float minStretch = px(80);
        float fixedScale = 1.0f;
        if (fixed > 0.0f && fixed + minStretch * static_cast<float>(stretch) > room)
            fixedScale = std::max(0.1f, (room - minStretch * static_cast<float>(stretch)) / fixed);
        const float rest = std::max(0.0f, room - fixed * fixedScale);
        float x = origin.x;
        for (std::size_t i = 0; i < cellsOf.size(); ++i) {
            const Cell& c = cellsOf[i];
            float w = c.fixedWidth > 0.0f ? px(c.fixedWidth) * fixedScale
                                          : (stretch > 0 ? rest * static_cast<float>(std::max(1, c.stretch)) / static_cast<float>(stretch) : 0.0f);
            float x1 = theme::snap(x + w);
            if (i + 1 == cellsOf.size()) x1 = origin.x + avail.x;   // the last cell takes the rounding
            if (x1 - x >= 1.0f) {
                ImGui::PushID(static_cast<int>(i));
                const cells::Rect content = cells::beginCell("##cell", ImVec2(x, origin.y), ImVec2(x1, origin.y + avail.y), c.title, c.meta);
                if (c.content && content.height() >= 1.0f) c.content(content);
                cells::endCell();
                ImGui::PopID();
            }
            x = x1 + gap;
        }
        ImGui::SetCursorScreenPos(origin);
        ImGui::Dummy(avail);
    }

} // namespace sirius::app::gui
