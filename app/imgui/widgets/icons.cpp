#include "imgui/widgets/icons.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>

namespace sirius::app::gui {

    namespace {

        constexpr float kPi = 3.14159265358979323846f;

        // Draws on the 24 x 24 grid, mapped into the target square.
        struct Pen {
            ImDrawList* dl;
            ImVec2 origin;
            float scale;
            ImU32 colour;
            float stroke;

            ImVec2 at(float x, float y) const { return ImVec2(origin.x + x * scale, origin.y + y * scale); }

            void line(float x1, float y1, float x2, float y2) const { dl->AddLine(at(x1, y1), at(x2, y2), colour, stroke); }
            void poly(std::initializer_list<ImVec2> pts, bool close = false) const {
                dl->PathClear();
                for (const ImVec2& p : pts) dl->PathLineTo(at(p.x, p.y));
                dl->PathStroke(colour, close ? ImDrawFlags_Closed : ImDrawFlags_None, stroke);
            }
            void box(float x, float y, float w, float h) const {
                dl->AddRect(at(x, y), at(x + w, y + h), colour, 0.0f, ImDrawFlags_None, stroke);
            }
            void ring(float cx, float cy, float r) const { dl->AddCircle(at(cx, cy), r * scale, colour, 0, stroke); }
            void ellipse(float cx, float cy, float rx, float ry) const {
                dl->AddEllipse(at(cx, cy), ImVec2(rx * scale, ry * scale), colour, 0.0f, 0, stroke);
            }
            void dot(float cx, float cy, float r) const { dl->AddCircleFilled(at(cx, cy), std::max(r * scale, 0.8f), colour); }
            void slab(float x, float y, float w, float h) const { dl->AddRectFilled(at(x, y), at(x + w, y + h), colour); }
            void wedge(std::initializer_list<ImVec2> pts) const {
                dl->PathClear();
                for (const ImVec2& p : pts) dl->PathLineTo(at(p.x, p.y));
                dl->PathFillConvex(colour);
            }
            void moveTo(float x, float y) const {
                dl->PathClear();
                dl->PathLineTo(at(x, y));
            }
            void cubicTo(float c1x, float c1y, float c2x, float c2y, float x, float y) const {
                dl->PathBezierCubicCurveTo(at(c1x, c1y), at(c2x, c2y), at(x, y));
            }
            void strokePath(bool close = false) const {
                dl->PathStroke(colour, close ? ImDrawFlags_Closed : ImDrawFlags_None, stroke);
            }
        };

        void draw(const Pen& p, Icon icon) {
            switch (icon) {
                case Icon::None: break;
                case Icon::Navigate:   // four-way move
                    p.line(12, 3, 12, 21);
                    p.line(3, 12, 21, 12);
                    p.poly({{9, 6}, {12, 3}, {15, 6}});
                    p.poly({{9, 18}, {12, 21}, {15, 18}});
                    p.poly({{6, 9}, {3, 12}, {6, 15}});
                    p.poly({{18, 9}, {21, 12}, {18, 15}});
                    break;
                case Icon::Probe:   // crosshair
                    p.line(12, 2, 12, 22);
                    p.line(2, 12, 22, 12);
                    p.ring(12, 12, 4.5f);
                    break;
                case Icon::Measure:   // double-headed ruler arrow
                    p.line(3, 12, 21, 12);
                    p.poly({{7, 8}, {3, 12}, {7, 16}});
                    p.poly({{17, 8}, {21, 12}, {17, 16}});
                    break;
                case Icon::Roi:   // selection box with corner handles
                    p.box(4, 4, 16, 16);
                    p.slab(2.6f, 2.6f, 2.8f, 2.8f);
                    p.slab(18.6f, 2.6f, 2.8f, 2.8f);
                    p.slab(2.6f, 18.6f, 2.8f, 2.8f);
                    p.slab(18.6f, 18.6f, 2.8f, 2.8f);
                    break;
                case Icon::Prompt:   // a pointer aimed at an object
                    p.poly({{5, 4}, {5, 18}, {8.8f, 14.6f}, {11.6f, 20.6f}, {14, 19.5f}, {11.3f, 13.6f}, {16.2f, 13.6f}}, true);
                    p.ring(18.5f, 5.5f, 3.2f);
                    break;
                case Icon::Brush:
                    p.dot(8, 16, 3.8f);
                    p.line(10.8f, 13.2f, 20, 4);
                    break;
                case Icon::Erase:
                    p.poly({{3, 13}, {12, 4}, {19, 11}, {10, 20}}, true);
                    p.line(9, 20.5f, 21, 20.5f);
                    break;
                case Icon::Fill:   // filled region inside a frame
                    p.box(3.5f, 3.5f, 17, 17);
                    p.slab(8, 8, 8, 8);
                    break;
                case Icon::Pick:   // target
                    p.ring(12, 12, 8);
                    p.dot(12, 12, 3);
                    break;
                case Icon::Merge:   // two overlapping labels
                    p.ring(9, 12, 6);
                    p.ring(15, 12, 6);
                    break;
                case Icon::Split:   // one label forking into two
                    p.line(12, 21, 12, 12);
                    p.line(12, 12, 5, 5);
                    p.line(12, 12, 19, 5);
                    p.poly({{5, 9}, {5, 5}, {9, 5}});
                    p.poly({{15, 5}, {19, 5}, {19, 9}});
                    break;
                case Icon::Lasso:
                    p.ellipse(12, 10, 8.5f, 6.0f);
                    p.line(9, 15.5f, 8, 19.5f);
                    p.dot(7.5f, 20.5f, 1.6f);
                    break;
                case Icon::Plus:
                    p.line(12, 5, 12, 19);
                    p.line(5, 12, 19, 12);
                    break;
                case Icon::Minus: p.line(5, 12, 19, 12); break;
                case Icon::ZoomIn:
                    p.ring(10.5f, 10.5f, 7);
                    p.line(15.5f, 15.5f, 21, 21);
                    p.line(10.5f, 7, 10.5f, 14);
                    p.line(7, 10.5f, 14, 10.5f);
                    break;
                case Icon::ZoomOut:
                    p.ring(10.5f, 10.5f, 7);
                    p.line(15.5f, 15.5f, 21, 21);
                    p.line(7, 10.5f, 14, 10.5f);
                    break;
                case Icon::Fit:   // arrows to opposite corners
                    p.poly({{14, 3}, {21, 3}, {21, 10}});
                    p.line(21, 3, 14, 10);
                    p.poly({{10, 21}, {3, 21}, {3, 14}});
                    p.line(3, 21, 10, 14);
                    break;
                case Icon::Eye:
                    p.moveTo(2, 12);
                    p.cubicTo(5, 6.5f, 8.5f, 5, 12, 5);
                    p.cubicTo(15.5f, 5, 19, 6.5f, 22, 12);
                    p.cubicTo(19, 17.5f, 15.5f, 19, 12, 19);
                    p.cubicTo(8.5f, 19, 5, 17.5f, 2, 12);
                    p.strokePath(true);
                    p.dot(12, 12, 2.8f);
                    break;
                case Icon::Pin:   // the pinned Load step's hexagon
                    p.wedge({{12, 2}, {21, 7}, {21, 17}, {12, 22}, {3, 17}, {3, 7}});
                    break;
                case Icon::Play: p.wedge({{7, 4}, {20, 12}, {7, 20}}); break;
                case Icon::Pause:
                    p.slab(7, 4, 3.6f, 16);
                    p.slab(13.4f, 4, 3.6f, 16);
                    break;
                case Icon::Maximize:   // corner brackets
                    p.poly({{3, 9}, {3, 3}, {9, 3}});
                    p.poly({{15, 3}, {21, 3}, {21, 9}});
                    p.poly({{21, 15}, {21, 21}, {15, 21}});
                    p.poly({{9, 21}, {3, 21}, {3, 15}});
                    break;
                case Icon::Float:   // a window lifted off another
                    p.box(2.5f, 2.5f, 13, 13);
                    p.box(8.5f, 8.5f, 13, 13);
                    break;
                case Icon::Dock:   // panel pinned to the bottom edge
                    p.box(3, 4, 18, 16);
                    p.slab(4.5f, 14.5f, 15, 4);
                    break;
                case Icon::ChevronUp: p.poly({{6, 15}, {12, 9}, {18, 15}}); break;
                case Icon::ChevronDown: p.poly({{6, 9}, {12, 15}, {18, 9}}); break;
                case Icon::ChevronRight: p.poly({{9, 6}, {15, 12}, {9, 18}}); break;
                case Icon::Trash:
                    p.line(4, 7.5f, 20, 7.5f);
                    p.line(9.5f, 4.5f, 14.5f, 4.5f);
                    p.box(6.5f, 7.5f, 11, 12);
                    p.line(10, 11, 10, 16);
                    p.line(14, 11, 14, 16);
                    break;
                case Icon::Sparkle: {   // four-point star with concave arms
                    // four concave quadrants around the centre, each filled as a fan
                    static const ImVec2 tips[5] = {{12, 2}, {22, 12}, {12, 22}, {2, 12}, {12, 2}};
                    for (int q = 0; q < 4; ++q) {
                        p.dl->PathClear();
                        p.dl->PathLineTo(p.at(12, 12));
                        p.dl->PathLineTo(p.at(tips[q].x, tips[q].y));
                        p.dl->PathBezierQuadraticCurveTo(p.at(12, 12), p.at(tips[q + 1].x, tips[q + 1].y), 12);
                        // the curve hugs the centre, so the fan from the centre is thin and convex enough
                        p.dl->PathFillConcave(p.colour);
                    }
                    break;
                }
                case Icon::Pencil:
                    p.poly({{3, 21}, {4.5f, 15.8f}, {16, 4.3f}, {19.7f, 8}, {8.2f, 19.5f}}, true);
                    p.line(14.4f, 5.9f, 18.1f, 9.6f);
                    break;
                case Icon::Info:
                    p.ring(12, 12, 9);
                    p.line(12, 11, 12, 17);
                    p.dot(12, 7.4f, 1.2f);
                    break;
                case Icon::Check: p.poly({{4, 12.5f}, {9.5f, 18}, {20, 6}}); break;
                case Icon::Close:
                    p.line(5, 5, 19, 19);
                    p.line(19, 5, 5, 19);
                    break;
                case Icon::Enter:   // the return key of the assistant's input
                    p.poly({{20, 5}, {20, 15}, {5, 15}});
                    p.poly({{10, 10}, {5, 15}, {10, 20}});
                    break;
                case Icon::Recompute: {   // the Recompute cache policy
                    // the arc of the circle in (4, 4, 16, 16) from 65 degrees, sweeping -295
                    // (clockwise on screen); angles are counter-clockwise with y up.
                    const float a0 = 65.0f * kPi / 180.0f, a1 = (65.0f - 295.0f) * kPi / 180.0f;
                    p.dl->PathClear();
                    const int n = 28;
                    for (int i = 0; i <= n; ++i) {
                        const float a = a0 + (a1 - a0) * static_cast<float>(i) / static_cast<float>(n);
                        p.dl->PathLineTo(p.at(12 + 8 * std::cos(a), 12 - 8 * std::sin(a)));
                    }
                    p.strokePath();
                    p.poly({{14.5f, 3.5f}, {19.5f, 6.4f}, {14.5f, 9.3f}});
                    break;
                }
                case Icon::More:   // the overflow "..."
                    p.dot(5.5f, 12, 1.6f);
                    p.dot(12, 12, 1.6f);
                    p.dot(18.5f, 12, 1.6f);
                    break;
                case Icon::Server:   // two stacked units, a light on each
                    p.box(4, 4, 16, 7);
                    p.box(4, 13, 16, 7);
                    p.dot(7.5f, 7.5f, 1.2f);
                    p.dot(7.5f, 16.5f, 1.2f);
                    p.line(11.5f, 7.5f, 16.5f, 7.5f);
                    p.line(11.5f, 16.5f, 16.5f, 16.5f);
                    break;
                case Icon::Copy:   // the copy in front, the original behind it
                    p.box(4, 8.5f, 11.5f, 11.5f);
                    p.poly({{8.5f, 8.5f}, {8.5f, 4}, {20, 4}, {20, 15.5f}, {15.5f, 15.5f}});
                    break;
                case Icon::Help:
                    p.ring(12, 12, 9);
                    p.moveTo(8.6f, 9.4f);
                    p.cubicTo(9.2f, 6.4f, 14.8f, 6.2f, 15.2f, 9.4f);
                    p.cubicTo(15.5f, 11.8f, 12, 12.2f, 12, 15);
                    p.strokePath();
                    p.dot(12, 18, 1.2f);
                    break;
            }
        }

    } // namespace

    void drawIcon(ImDrawList* dl, ImVec2 min, ImVec2 max, Icon icon, ImU32 colour, float strokePx) {
        if (icon == Icon::None || !dl) return;
        const float w = max.x - min.x, h = max.y - min.y;
        if (w <= 0.0f || h <= 0.0f) return;
        const float side = std::min(w, h);
        Pen pen;
        pen.dl = dl;
        pen.origin = ImVec2((min.x + max.x - side) * 0.5f, (min.y + max.y - side) * 0.5f);
        pen.scale = side / 24.0f;
        pen.colour = colour;
        pen.stroke = strokePx;
        const ImDrawListFlags saved = dl->Flags;
        dl->Flags |= ImDrawListFlags_AntiAliasedLines | ImDrawListFlags_AntiAliasedFill;
        draw(pen, icon);
        dl->Flags = saved;
    }

    void drawIcon(ImDrawList* dl, ImVec2 centre, float side, Icon icon, ImU32 colour, float strokePx) {
        drawIcon(dl, ImVec2(centre.x - side * 0.5f, centre.y - side * 0.5f),
                 ImVec2(centre.x + side * 0.5f, centre.y + side * 0.5f), icon, colour, strokePx);
    }

} // namespace sirius::app::gui
