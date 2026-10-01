#include "imgui/viewer/slice_pane.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "imgui/strings.hpp"
#include "imgui/viewer/trace.hpp"
#include "imgui/viewer/viewer_constants.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    using theme::px;

    namespace {
        float lineHeight(float designPx) { return theme::textSize("Ag", designPx).y; }

        // A dashed straight segment: dashes of 4 pens, gaps of 2.
        void dashedLine(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 color, float thickness) {
            dashedPolyline(dl, {a, b}, color, thickness, 4.0f * thickness, 2.0f * thickness);
        }
    } // namespace

    float drawOverlayText(ImDrawList* dl, ImVec2 pos, const std::string& text, bool bold, float opacity, float designPx) {
        const theme::Weight w = bold ? theme::Weight::ExtraBold : theme::Weight::Regular;
        widgets::drawText(dl, pos, text, designPx, theme::withAlpha(theme::kViewerText, opacity), w);
        return theme::textSize(text, designPx, w).x;
    }

    void drawCenteredWrapped(ImDrawList* dl, ImVec2 min, ImVec2 max, const std::string& text, float designPx, ImU32 color) {
        if (text.empty()) return;
        const float room = std::max(1.0f, max.x - min.x);
        std::vector<std::string> lines;
        for (const std::string& paragraph : split(text, '\n')) {
            std::string line;
            for (const std::string& word : split(paragraph, ' ', true)) {
                const std::string candidate = line.empty() ? word : line + " " + word;
                if (!line.empty() && theme::textSize(candidate, designPx).x > room) {
                    lines.push_back(line);
                    line = word;
                } else {
                    line = candidate;
                }
            }
            lines.push_back(line);
        }
        const float lh = lineHeight(designPx);
        const float total = lh * static_cast<float>(lines.size());
        float y = min.y + (max.y - min.y - total) * 0.5f;
        for (const std::string& l : lines) {
            widgets::drawTextIn(dl, ImVec2(min.x, y), ImVec2(max.x, y + lh), l, designPx, color);
            y += lh;
        }
    }

    void dashedOutline(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 color, float thickness) {
        dashedPolyline(dl, {a, ImVec2(b.x, a.y), b, ImVec2(a.x, b.y), a}, color, thickness, 4.0f * thickness, 2.0f * thickness);
    }

    SlicePane::SlicePane(Kind kind, std::string name) : kind_(kind), name_(std::move(name)) {}

    void SlicePane::setContent(const Image& img, int factor, Index cols, Index rows, int originX, int originY) {
        factor_ = std::max(factor, 1);
        cols_ = cols;
        rows_ = rows;
        originX_ = originX;
        originY_ = originY;
        if (img.isNull()) {
            hasImage_ = false;
            return;
        }
        const bool smooth = smooth_ || view_.zx * factor_ < 1.0;
        texture_.upload(img.bytes(), img.width, img.height, smooth);
        imageW_ = img.width;
        imageH_ = img.height;
        hasImage_ = true;
    }

    void SlicePane::clearContent() {
        hasImage_ = false;
        cols_ = rows_ = 0;
    }

    SlicePane::View SlicePane::fitView(double ax, double ay) const {
        View v;
        if (cols_ <= 0 || rows_ <= 0) return v;
        const double ex = static_cast<double>(cols_) * ax, ey = static_cast<double>(rows_) * ay;
        const double z = std::max(1e-6, std::min(width() / ex, height() / ey));
        v.zx = z * ax;
        v.zy = z * ay;
        v.ox = (width() - static_cast<double>(cols_) * v.zx) / 2.0;
        v.oy = (height() - static_cast<double>(rows_) * v.zy) / 2.0;
        return v;
    }

    DPoint SlicePane::toVoxel(const DPoint& s) const { return {(s.x - view_.ox) / view_.zx, (s.y - view_.oy) / view_.zy}; }

    DPoint SlicePane::toScreen(const DPoint& v) const { return {view_.ox + v.x * view_.zx, view_.oy + v.y * view_.zy}; }

    ImVec2 SlicePane::toScreenAbs(const DPoint& v) const {
        const DPoint s = toScreen(v);
        return ImVec2(min_.x + static_cast<float>(s.x), min_.y + static_cast<float>(s.y));
    }

    bool SlicePane::inside(const DPoint& v) const {
        return v.x >= 0.0 && v.y >= 0.0 && v.x < static_cast<double>(cols_) && v.y < static_cast<double>(rows_);
    }

    bool SlicePane::place(ImVec2 min, ImVec2 max) {
        const bool resized = std::abs((max.x - min.x) - (max_.x - min_.x)) > 0.5f || std::abs((max.y - min.y) - (max_.y - min_.y)) > 0.5f;
        min_ = min;
        max_ = max;
        return resized;
    }

    // --- input -------------------------------------------------------------------------

    bool SlicePane::input() {
        const ImGuiIO& io = ImGui::GetIO();
        ImGui::SetCursorScreenPos(min_);
        ImGui::SetNextItemAllowOverlap();
        ImGui::InvisibleButton(name_.c_str(), ImVec2(std::max(1.0f, max_.x - min_.x), std::max(1.0f, max_.y - min_.y)),
                               ImGuiButtonFlags_MouseButtonMask_);
        hovered_ = ImGui::IsItemHovered();
        const bool activated = ImGui::IsItemActivated();
        const DPoint local(static_cast<double>(io.MousePos.x - min_.x), static_cast<double>(io.MousePos.y - min_.y));
        const bool moved = io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f;
        const ImGuiKeyChord mods = io.KeyMods;
        bool pressed = false;

        if (activated) {
            int b = -1;
            for (int i = 0; i < 3 && b < 0; ++i)
                if (ImGui::IsMouseClicked(i)) b = i;
            mouse_ = local;
            if (b == ImGuiMouseButton_Right) {
                if (onContextMenu) onContextMenu(io.MousePos, toVoxel(local));
            } else if (b == ImGuiMouseButton_Left && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                // the second press of a double-click is the double-click, not a press
                releaseAfterDouble_ = true;
                if (onDoubleClick) onDoubleClick(toVoxel(local), mods);
                pressed = true;
            } else if (b >= 0) {
                synthPress(local, b);
                pressed = true;
            }
        }
        // drag, hover: every move over the pane, or anywhere while a button holds it
        if (button_ >= 0 && ImGui::IsMouseDown(button_) && moved) {
            synthMove(local);
        } else if ((hovered_ || button_ >= 0) && (moved || !mouseIn_)) {
            mouse_ = local;
            mouseIn_ = true;
            if (onHover) onHover(toVoxel(local));
        }
        if (button_ >= 0 && !ImGui::IsMouseDown(button_)) synthRelease(local);
        if (releaseAfterDouble_ && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            releaseAfterDouble_ = false;
            if (onRelease) onRelease(toVoxel(local), ImGuiMouseButton_Left, mods, moved_);
        }
        if (hovered_ && io.MouseWheel != 0.0f && onWheel) onWheel(local, static_cast<double>(io.MouseWheel), mods);
        if (mouseIn_ && !hovered_ && button_ < 0) {
            mouseIn_ = false;
            if (onExit) onExit();
        }
        if (hovered_ || button_ >= 0) ImGui::SetMouseCursor(button_ >= 0 ? cursorHeld_ : cursor_);
        return pressed;
    }

    void SlicePane::synthPress(const DPoint& local, int button) {
        mouse_ = local;
        mouseIn_ = true;
        button_ = button;
        pressPos_ = lastDrag_ = local;
        moved_ = false;
        if (onPress) onPress(toVoxel(local), button, ImGui::GetIO().KeyMods);
    }

    void SlicePane::synthMove(const DPoint& local) {
        mouse_ = local;
        mouseIn_ = true;
        if (button_ >= 0) {
            const DPoint delta = local - lastDrag_;
            lastDrag_ = local;
            if (std::abs(local.x - pressPos_.x) + std::abs(local.y - pressPos_.y) > 2.0) moved_ = true;
            if (onDrag) onDrag(toVoxel(local), delta, button_, ImGui::GetIO().KeyMods);
        }
        if (onHover) onHover(toVoxel(local));
    }

    void SlicePane::synthRelease(const DPoint& local) {
        mouse_ = local;
        const int b = button_;
        button_ = -1;
        if (b >= 0 && onRelease) onRelease(toVoxel(local), b, ImGui::GetIO().KeyMods, moved_);
    }

    void SlicePane::keyNavigation() {
        if (!onKeyNavigate) return;
        const ImGuiIO& io = ImGui::GetIO();
        // IsKeyPressed ignores modifiers, and an arrow with Ctrl, Alt or
        // Super is a window action (Alt+Up moves the step): only the plain
        // and Shift+ keys walk the crosshair.
        if (io.KeyCtrl || io.KeyAlt || io.KeySuper) return;
        const int step = io.KeyShift ? 10 : 1;
        int dc = 0, dr = 0, dd = 0;
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) dc = -step;
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) dc = step;
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) dr = -step;
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) dr = step;
        if (ImGui::IsKeyPressed(ImGuiKey_PageUp)) dd = -step;
        if (ImGui::IsKeyPressed(ImGuiKey_PageDown)) dd = step;
        if (dc || dr || dd) onKeyNavigate(dc, dr, dd);
    }

    // --- drawing --------------------------------------------------------------------------

    void SlicePane::draw(ImDrawList* dl) {
        if (!placed()) return;
        const bool trace = ScopedTrace::enabled();
        TraceClock clock;
        const float w = max_.x - min_.x, h = max_.y - min_.y;
        dl->PushClipRect(min_, max_, true);
        dl->AddRectFilled(min_, max_, theme::kViewerGround);

        if (hasImage_ && cols_ > 0 && rows_ > 0 && texture_.valid()) {
            // nearest neighbour when magnifying keeps voxels crisp; smooth
            // when shrinking avoids aliasing on large frames
            const bool smooth = smooth_ || view_.zx * factor_ < 1.0;
            texture_.setSmooth(smooth);
            const ImVec2 a(min_.x + static_cast<float>(view_.ox + originX_ * view_.zx), min_.y + static_cast<float>(view_.oy + originY_ * view_.zy));
            const ImVec2 b(a.x + static_cast<float>(imageW_ * factor_ * view_.zx), a.y + static_cast<float>(imageH_ * factor_ * view_.zy));
            texture_.draw(dl, a, b);
        }

        if (tracks_ && hasContent())
            paintTrackPaths(dl, *tracks_, trackOptions_, [this](const DPoint& v) { return toScreenAbs(v); }, min_, max_);

        // annotations (ROI boxes dashed, measurements in accent) sit under the crosshair
        if (hasContent()) {
            for (const Annotation& a : annotations_) {
                const float opacity = a.pending ? 0.6f : 1.0f;
                if (a.kind == Annotation::Kind::Roi) {
                    if (a.rect.isNull()) continue;
                    const ImVec2 r0 = toScreenAbs(DPoint(a.rect.x0, a.rect.y0)), r1 = toScreenAbs(DPoint(a.rect.x1, a.rect.y1));
                    dashedOutline(dl, r0, r1, theme::withAlpha(theme::kViewerText, opacity), px(1.0f));
                    if (!a.text.empty()) drawOverlayText(dl, ImVec2(r0.x + px(4), r0.y - px(16)), a.text, true, opacity);
                } else {
                    if (a.points.empty()) continue;
                    const ImU32 c = theme::withAlpha(theme::kAccent, opacity);
                    const float pen = px(1.5f);
                    for (std::size_t i = 0; i + 1 < a.points.size(); ++i) dl->AddLine(toScreenAbs(a.points[i]), toScreenAbs(a.points[i + 1]), c, pen);
                    for (const DPoint& v : a.points) {
                        const ImVec2 sp = toScreenAbs(v);
                        dl->AddLine(ImVec2(sp.x - px(4), sp.y), ImVec2(sp.x + px(4), sp.y), c, pen);
                        dl->AddLine(ImVec2(sp.x, sp.y - px(4)), ImVec2(sp.x, sp.y + px(4)), c, pen);
                    }
                    if (!a.text.empty()) {
                        const ImVec2 last = toScreenAbs(a.points.back());
                        drawOverlayText(dl, ImVec2(last.x + px(8), last.y - px(16)), a.text, true, opacity);
                    }
                }
            }
        }

        if (crossVisible_ && hasContent()) {
            const ImVec2 c = toScreenAbs(cross_ + DPoint(0.5, 0.5));
            const float pen = theme::crispPen(1.0f);
            const float x = theme::snap(c.x), y = theme::snap(c.y);
            if (crossLocked_) {
                // locked: dashed, at 45 %
                const ImU32 col = theme::withAlpha(theme::kAccent, 0.45f);
                dashedLine(dl, ImVec2(x + pen * 0.5f, min_.y), ImVec2(x + pen * 0.5f, max_.y), col, pen);
                dashedLine(dl, ImVec2(min_.x, y + pen * 0.5f), ImVec2(max_.x, y + pen * 0.5f), col, pen);
            } else {
                dl->AddRectFilled(ImVec2(x, min_.y), ImVec2(x + pen, max_.y), theme::kAccent);
                dl->AddRectFilled(ImVec2(min_.x, y), ImVec2(max_.x, y + pen), theme::kAccent);
            }
        }

        // prompts above the crosshair: they are what the clicks act on
        if (hasContent() && !promptMarks_.empty()) {
            const float pen = px(1.5f);
            const ImU32 edge = theme::withAlpha(theme::kViewerGround, 0.85f);
            const auto centre = [this](const DPoint& v) { return toScreenAbs(v + DPoint(0.5, 0.5)); };
            // the faint projections first, so nothing on the plane is under one
            for (const bool onPlane : {false, true})
                for (const PromptMark& m : promptMarks_) {
                    if ((m.inPlane || m.pending) != onPlane) continue;
                    const ImU32 ink = m.object ? theme::kAccent : theme::kViewerText;
                    switch (m.shape) {
                        case PromptMark::Shape::Point: {
                            const ImVec2 c = centre(m.a);
                            if (!m.inPlane) {
                                const float r = px(3.0f);
                                dl->AddCircle(c, r, theme::withAlpha(theme::kViewerGround, 0.5f), 0, px(3.0f));
                                dl->AddCircle(c, r, theme::withAlpha(ink, 0.6f), 0, pen);
                                break;
                            }
                            const float r = px(5.5f), ring = px(1.5f);
                            dl->AddCircleFilled(c, r + ring + px(1.0f), edge);
                            dl->AddCircleFilled(c, r + ring, theme::kViewerText);
                            dl->AddCircleFilled(c, r, m.object ? theme::kAccent : theme::kViewerGround);
                            const float arm = r * 0.55f;
                            dl->AddLine(ImVec2(c.x - arm, c.y), ImVec2(c.x + arm, c.y), theme::kViewerText, pen);
                            if (m.object) dl->AddLine(ImVec2(c.x, c.y - arm), ImVec2(c.x, c.y + arm), theme::kViewerText, pen);
                            break;
                        }
                        case PromptMark::Shape::Box: {
                            const ImVec2 r0 = toScreenAbs(m.a), r1 = toScreenAbs(m.b);
                            if (m.pending) {
                                dl->AddRect(r0, r1, edge, 0.0f, ImDrawFlags_None, px(3.0f));
                                dashedOutline(dl, r0, r1, theme::kViewerText, pen);
                            } else if (!m.inPlane) {
                                dashedOutline(dl, r0, r1, theme::withAlpha(theme::kAccent, 0.55f), px(1.0f));
                            } else {
                                dl->AddRect(r0, r1, edge, 0.0f, ImDrawFlags_None, px(4.0f));
                                dl->AddRect(r0, r1, theme::kAccent, 0.0f, ImDrawFlags_None, px(2.0f));
                            }
                            break;
                        }
                        case PromptMark::Shape::Stroke: {
                            if (m.stroke.empty()) break;
                            std::vector<ImVec2> line;
                            for (const DPoint& v : m.stroke) line.push_back(centre(v));
                            if (line.size() == 1) line.push_back(ImVec2(line[0].x + 0.5f, line[0].y));
                            const int n = static_cast<int>(line.size());
                            if (!m.inPlane && !m.pending) {
                                dl->AddPolyline(line.data(), n, theme::withAlpha(ink, 0.5f), ImDrawFlags_None, px(1.0f));
                                break;
                            }
                            dl->AddPolyline(line.data(), n, edge, ImDrawFlags_None, px(4.5f));
                            dl->AddPolyline(line.data(), n, m.pending ? theme::kViewerText : ink, ImDrawFlags_None, px(2.0f));
                            break;
                        }
                    }
                }
        }

        if (brush_ && mouseIn_ && hasContent()) {
            const float r = static_cast<float>(std::max(1.0, brushRadius_ * view_.zx));
            const float ry = r * static_cast<float>(view_.zy / std::max(1e-9, view_.zx));
            dl->AddEllipse(ImVec2(min_.x + static_cast<float>(mouse_.x), min_.y + static_cast<float>(mouse_.y)), ImVec2(r, ry), theme::kViewerText, 0.0f,
                           0, px(1.5f));
        }

        // corner label, scale bar, hint
        if (!title_.empty()) {
            // bold first token, the rest at 70 %
            const std::size_t cut = title_.find("  ");
            const std::string head = cut == std::string::npos ? title_ : title_.substr(0, cut);
            const ImVec2 at(min_.x + px(static_cast<float>(viewer::kOverlayInset)), min_.y + px(static_cast<float>(viewer::kOverlayTop)));
            const float headW = drawOverlayText(dl, at, head, true);
            if (cut != std::string::npos)
                drawOverlayText(dl, ImVec2(at.x + headW + px(static_cast<float>(viewer::kOverlayGap)), at.y), title_.substr(cut + 2), false, 0.7f);
        }
        const float fmH = lineHeight(11);
        if (umPerVoxel_ > 0.0 && hasContent()) {
            // largest of the design's steps that stays under 140 px
            static const double steps[] = {50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05};
            double um = steps[sizeof steps / sizeof steps[0] - 1];
            for (double s : steps) {
                if (s / umPerVoxel_ * view_.zx <= static_cast<double>(px(static_cast<float>(viewer::kScaleBarMaxPx)))) {
                    um = s;
                    break;
                }
            }
            const float barPx = static_cast<float>(um / umPerVoxel_ * view_.zx);
            const std::string label = um >= 1.0 ? format("%g \xC2\xB5m", um) : format("%g nm", um * 1000.0);
            const float lw = theme::textSize(label, 11).x;
            const float inset = px(static_cast<float>(viewer::kOverlayInset)), bottom = px(static_cast<float>(viewer::kOverlayBottom));
            const float x1 = max_.x - inset - lw - px(6);
            const float barY = theme::snap(max_.y - bottom - px(8));
            dl->AddRectFilled(ImVec2(theme::snap(x1 - barPx), barY), ImVec2(theme::snap(x1), barY + theme::crispPen(2.0f)), theme::kViewerText);
            drawOverlayText(dl, ImVec2(max_.x - inset - lw, max_.y - bottom - fmH), label);
        }
        if (!hint_.empty())
            drawOverlayText(dl, ImVec2(min_.x + px(static_cast<float>(viewer::kOverlayInset)), max_.y - px(static_cast<float>(viewer::kOverlayBottom)) - fmH),
                            hint_, false, 0.75f);
        if (!message_.empty())
            drawCenteredWrapped(dl, ImVec2(min_.x + px(12), min_.y + px(12)), ImVec2(max_.x - px(12), max_.y - px(12)), message_, 12,
                                theme::rgb(243, 242, 242, 180));
        dl->PopClipRect();
        if (trace)
            std::fprintf(stderr, "pane %s paint %lld us (%.0fx%.0f, image %dx%d)\n", name_.c_str(), clock.micros(), static_cast<double>(w),
                         static_cast<double>(h), hasImage_ ? imageW_ : 0, hasImage_ ? imageH_ : 0);
    }

} // namespace sirius::app::gui
