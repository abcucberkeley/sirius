#include "imgui/theme.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <implot.h>

#include "core/app_paths.hpp"
#include "imgui/strings.hpp"

namespace sirius::app::gui::theme {

    namespace {
        float gScale = 1.0f;
        ImFont* gFonts[4] = {nullptr, nullptr, nullptr, nullptr};
        ImFont* gMono = nullptr;
        ImFont* gCaption = nullptr;

        // The first existing file of `candidates`, "" when none is there.
        std::string firstExisting(const std::vector<std::string>& candidates) {
            for (const std::string& c : candidates)
                if (isFile(c)) return c;
            return std::string();
        }

        std::string fontDirectory() {
            std::vector<std::string> dirs;
            for (const std::string& d : {besideApplication("fonts"), installedDataDirectory("fonts")})
                if (!d.empty()) dirs.push_back(d);
#ifdef SIRIUS_APP_SOURCE_DIR
            dirs.push_back(std::string(SIRIUS_APP_SOURCE_DIR) + "/resources/fonts");
#endif
            for (const std::string& d : dirs)
                if (isFile(d + "/Archivo-Regular.ttf")) return d;
            return std::string();
        }

        // Archivo is Latin only, and operation labels carry Greek ("Emission
        // λ"), arrows and maths. Glyphs are looked up through the merged
        // sources in order, so the rest of the text still comes out in Archivo.
        std::vector<std::string> fallbackFaces(bool bold) {
            std::vector<std::string> out;
#ifdef _WIN32
            const std::string dir = "C:/Windows/Fonts/";
            out.push_back(firstExisting({dir + (bold ? "segoeuib.ttf" : "segoeui.ttf"), dir + "segoeui.ttf"}));
            out.push_back(firstExisting({dir + "seguisym.ttf"}));
#elif defined(__APPLE__)
            out.push_back(firstExisting({"/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Unicode.ttf"}));
            out.push_back(firstExisting({"/System/Library/Fonts/Apple Symbols.ttf"}));
#else
            (void)bold;
            out.push_back(firstExisting({"/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                                         "/usr/share/fonts/noto/NotoSans-Regular.ttf",
                                         "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
                                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                         "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                                         "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                                         "/usr/share/fonts/TTF/DejaVuSans.ttf",
                                         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"}));
            out.push_back(firstExisting({"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                         "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                                         "/usr/share/fonts/TTF/DejaVuSans.ttf",
                                         "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf"}));
#endif
            out.erase(std::remove(out.begin(), out.end(), std::string()), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            return out;
        }

        std::string monoFace() {
#ifdef _WIN32
            return firstExisting({"C:/Windows/Fonts/CascadiaMono.ttf", "C:/Windows/Fonts/consola.ttf"});
#elif defined(__APPLE__)
            return firstExisting({"/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"});
#else
            return firstExisting({"/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                                  "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
                                  "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
                                  "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
                                  "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                                  "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"});
#endif
        }

        ImFont* addFace(const std::string& file, const std::vector<std::string>& fallbacks, float tracking = 0.0f) {
            ImGuiIO& io = ImGui::GetIO();
            ImFont* f = nullptr;
            if (!file.empty()) {
                ImFontConfig cfg;
                cfg.GlyphExtraAdvanceX = tracking;
                f = io.Fonts->AddFontFromFileTTF(file.c_str(), kBodyPx, &cfg);
            }
            if (!f) f = io.Fonts->AddFontDefault();
            for (const std::string& fb : fallbacks) {
                ImFontConfig cfg;
                cfg.MergeMode = true;
                io.Fonts->AddFontFromFileTTF(fb.c_str(), kBodyPx, &cfg);
            }
            return f;
        }
    } // namespace

    ImU32 withAlpha(ImU32 c, float opacity) {
        const float a = static_cast<float>((c >> 24) & 0xFF) * std::clamp(opacity, 0.0f, 1.0f);
        return (c & 0x00FFFFFFu) | (static_cast<ImU32>(a + 0.5f) << 24);
    }

    ImU32 mix(ImU32 a, ImU32 b, float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        ImU32 out = 0;
        for (int shift = 0; shift < 32; shift += 8) {
            const float ca = static_cast<float>((a >> shift) & 0xFF), cb = static_cast<float>((b >> shift) & 0xFF);
            out |= static_cast<ImU32>(ca + (cb - ca) * t + 0.5f) << shift;
        }
        return out;
    }

    ImVec4 vec(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }

    ImU32 fromFloat(const std::array<float, 3>& c, float alpha) {
        const auto b = [](float v) { return static_cast<int>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
        return rgb(b(c[0]), b(c[1]), b(c[2]), b(alpha));
    }

    std::string hex(ImU32 c) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "#%02x%02x%02x", static_cast<int>(c & 0xFF), static_cast<int>((c >> 8) & 0xFF),
                      static_cast<int>((c >> 16) & 0xFF));
        return buf;
    }

    ImFont* font(Weight w) { return gFonts[static_cast<int>(w)]; }
    ImFont* mono() { return gMono; }
    ImFont* captionFont() { return gCaption ? gCaption : gFonts[0]; }

    FontScope::FontScope(float designPx, Weight w) { ImGui::PushFont(font(w), designPx); }
    FontScope::FontScope(float designPx, ImFont* face) { ImGui::PushFont(face, designPx); }
    FontScope::~FontScope() { ImGui::PopFont(); }

    ImVec2 textSize(const char* text, float designPx, Weight w, const char* end) {
        ImFont* f = font(w);
        if (!f) f = ImGui::GetFont();
        const ImGuiStyle& st = ImGui::GetStyle();
        const float size = designPx * st.FontScaleMain * st.FontScaleDpi;
        ImVec2 s = f->CalcTextSizeA(size, FLT_MAX, 0.0f, text, end);
        s.x = std::ceil(s.x);
        return s;
    }

    ImVec2 textSize(const std::string& text, float designPx, Weight w) {
        return textSize(text.c_str(), designPx, w, text.c_str() + text.size());
    }

    float scale() { return gScale; }

    void setScale(float s) { gScale = std::clamp(s, 0.5f, 4.0f); }

    float crispPen(float designPx) { return std::max(1.0f, std::round(designPx * gScale)); }

    float snap(float v) { return std::floor(v + 0.5f); }

    bool loadFonts() {
        const std::string dir = fontDirectory();
        static const char* faces[4] = {"Archivo-Regular.ttf", "Archivo-SemiBold.ttf", "Archivo-Bold.ttf",
                                       "Archivo-ExtraBold.ttf"};
        for (int i = 0; i < 4; ++i)
            gFonts[i] = addFace(dir.empty() ? std::string() : dir + "/" + faces[i], fallbackFaces(i >= 1));
        gMono = addFace(monoFace(), fallbackFaces(false));
        gCaption = addFace(dir.empty() ? std::string() : dir + "/" + faces[0], fallbackFaces(false), 1.0f);
        ImGui::GetIO().FontDefault = gFonts[0];
        return !dir.empty();
    }

    void applyTheme() {
        ImGuiStyle style;   // from the defaults, so re-applying at a new scale does not compound
        ImGui::StyleColorsLight(&style);

        // --- geometry: flat, square, ruled ---------------------------------
        style.WindowRounding = 0;
        style.ChildRounding = 0;
        style.FrameRounding = 0;
        style.PopupRounding = 0;
        style.ScrollbarRounding = 0;
        style.GrabRounding = 0;
        style.TabRounding = 0;
        style.WindowBorderSize = 0;
        style.ChildBorderSize = 0;
        style.PopupBorderSize = 2;
        style.FrameBorderSize = 1;          // 1.5 in the design: rounded to whole pixels in ScaleAllSizes terms
        style.TabBorderSize = 0;
        style.TabBarBorderSize = 2;
        style.TabBarOverlineSize = 0;
        style.DockingSeparatorSize = kRule;
        style.SeparatorSize = 1;
        style.WindowPadding = ImVec2(14, 12);
        style.FramePadding = ImVec2(8, 8);
        style.ItemSpacing = ImVec2(8, 8);
        style.ItemInnerSpacing = ImVec2(6, 4);
        style.CellPadding = ImVec2(6, 4);
        style.IndentSpacing = 16;
        style.ScrollbarSize = 10;
        style.ScrollbarPadding = 1;
        style.GrabMinSize = 10;
        style.WindowMenuButtonPosition = ImGuiDir_None;
        style.WindowTitleAlign = ImVec2(0.0f, 0.5f);
        style.ButtonTextAlign = ImVec2(0.0f, 0.5f);      // labels flush left
        style.SelectableTextAlign = ImVec2(0.0f, 0.5f);
        style.SeparatorTextBorderSize = 1;
        style.DisabledAlpha = 0.45f;
        style.DockingNodeHasCloseButton = false;
        style.TabCloseButtonMinWidthUnselected = FLT_MAX;
        style.HoverDelayNormal = 0.45f;
        style.HoverStationaryDelay = 0.12f;
        style.AntiAliasedLines = true;
        style.AntiAliasedFill = true;

        // --- colours -----------------------------------------------------------
        ImVec4* c = style.Colors;
        const ImVec4 none(0, 0, 0, 0);
        c[ImGuiCol_Text] = vec(kText);
        c[ImGuiCol_TextDisabled] = vec(kNeutral500);
        c[ImGuiCol_WindowBg] = vec(kBg);
        c[ImGuiCol_ChildBg] = none;
        c[ImGuiCol_PopupBg] = vec(kBg);
        c[ImGuiCol_Border] = vec(kDivider);
        c[ImGuiCol_BorderShadow] = none;
        c[ImGuiCol_FrameBg] = vec(kBg);
        c[ImGuiCol_FrameBgHovered] = vec(kBg);
        c[ImGuiCol_FrameBgActive] = vec(kBg);
        c[ImGuiCol_TitleBg] = vec(kSurface);
        c[ImGuiCol_TitleBgActive] = vec(kSurface);
        c[ImGuiCol_TitleBgCollapsed] = vec(kSurface);
        c[ImGuiCol_MenuBarBg] = vec(kBg);
        c[ImGuiCol_ScrollbarBg] = none;
        c[ImGuiCol_ScrollbarGrab] = vec(kNeutral400);
        c[ImGuiCol_ScrollbarGrabHovered] = vec(kNeutral500);
        c[ImGuiCol_ScrollbarGrabActive] = vec(kNeutral600);
        c[ImGuiCol_CheckMark] = vec(kAccent);
        c[ImGuiCol_SliderGrab] = vec(kAccent);
        c[ImGuiCol_SliderGrabActive] = vec(kAccent600);
        c[ImGuiCol_Button] = none;
        c[ImGuiCol_ButtonHovered] = vec(kNeutral200);
        c[ImGuiCol_ButtonActive] = vec(kNeutral300);
        c[ImGuiCol_Header] = vec(kSurface);
        c[ImGuiCol_HeaderHovered] = vec(kNeutral200);
        c[ImGuiCol_HeaderActive] = vec(kNeutral300);
        c[ImGuiCol_Separator] = vec(kDivider);
        c[ImGuiCol_SeparatorHovered] = vec(kAccent);
        c[ImGuiCol_SeparatorActive] = vec(kAccent);
        c[ImGuiCol_ResizeGrip] = none;
        c[ImGuiCol_ResizeGripHovered] = vec(kAccent);
        c[ImGuiCol_ResizeGripActive] = vec(kAccent600);
        c[ImGuiCol_InputTextCursor] = vec(kText);
        c[ImGuiCol_Tab] = none;
        c[ImGuiCol_TabHovered] = vec(kNeutral200);
        c[ImGuiCol_TabSelected] = vec(kBg);
        c[ImGuiCol_TabSelectedOverline] = vec(kAccent);
        c[ImGuiCol_TabDimmed] = none;
        c[ImGuiCol_TabDimmedSelected] = vec(kBg);
        c[ImGuiCol_TabDimmedSelectedOverline] = vec(kAccent);
        c[ImGuiCol_DockingPreview] = vec(withAlpha(kAccent, 0.35f));
        c[ImGuiCol_DockingEmptyBg] = vec(kNeutral900);
        c[ImGuiCol_PlotLines] = vec(kText);
        c[ImGuiCol_PlotLinesHovered] = vec(kAccent);
        c[ImGuiCol_PlotHistogram] = vec(kAccent);
        c[ImGuiCol_PlotHistogramHovered] = vec(kAccent600);
        c[ImGuiCol_TableHeaderBg] = none;
        c[ImGuiCol_TableBorderStrong] = vec(kDivider);
        c[ImGuiCol_TableBorderLight] = vec(kDivider);
        c[ImGuiCol_TableRowBg] = none;
        c[ImGuiCol_TableRowBgAlt] = none;
        c[ImGuiCol_TextLink] = vec(kAccentText);
        c[ImGuiCol_TextSelectedBg] = vec(withAlpha(kAccent, 0.35f));
        c[ImGuiCol_DragDropTarget] = vec(kAccent);
        c[ImGuiCol_NavCursor] = vec(kAccent);
        c[ImGuiCol_NavWindowingHighlight] = vec(kAccent);
        c[ImGuiCol_NavWindowingDimBg] = vec(withAlpha(kNeutral900, 0.2f));
        c[ImGuiCol_ModalWindowDimBg] = vec(withAlpha(kNeutral900, 0.35f));

        style.ScaleAllSizes(gScale);
        // borders stay whole pixels at fractional scales
        style.FrameBorderSize = crispPen(kBorder);
        style.PopupBorderSize = crispPen(2);
        style.TabBarBorderSize = crispPen(2);
        style.DockingSeparatorSize = crispPen(kRule);
        style.SeparatorSize = crispPen(1);
        style.FontSizeBase = kBodyPx;
        style.FontScaleMain = 1.0f;
        style.FontScaleDpi = gScale;
        ImGui::GetStyle() = style;

        if (ImPlot::GetCurrentContext()) {
            ImPlotStyle& ps = ImPlot::GetStyle();
            ps = ImPlotStyle();
            ps.Colors[ImPlotCol_FrameBg] = none;
            ps.Colors[ImPlotCol_PlotBg] = none;
            ps.Colors[ImPlotCol_PlotBorder] = none;
            ps.Colors[ImPlotCol_LegendBg] = vec(kBg);
            ps.Colors[ImPlotCol_LegendBorder] = vec(kDivider);
            ps.Colors[ImPlotCol_LegendText] = vec(kText);
            ps.Colors[ImPlotCol_TitleText] = vec(kText);
            ps.Colors[ImPlotCol_InlayText] = vec(kNeutral600);
            ps.Colors[ImPlotCol_AxisText] = vec(kNeutral600);
            ps.Colors[ImPlotCol_AxisGrid] = vec(withAlpha(kDivider, 0.5f));
            ps.Colors[ImPlotCol_AxisTick] = vec(kDivider);
            ps.Colors[ImPlotCol_AxisBg] = none;
            ps.Colors[ImPlotCol_AxisBgHovered] = none;
            ps.Colors[ImPlotCol_AxisBgActive] = none;
            ps.Colors[ImPlotCol_Selection] = vec(kAccent);
            ps.Colors[ImPlotCol_Crosshairs] = vec(kAccent);
            ps.PlotPadding = ImVec2(px(8), px(8));
            ps.LabelPadding = ImVec2(px(4), px(4));
            ps.PlotBorderSize = 0;
            ps.MinorAlpha = 0.0f;
            ps.PlotMinSize = ImVec2(px(60), px(40));
        }
    }

} // namespace sirius::app::gui::theme
