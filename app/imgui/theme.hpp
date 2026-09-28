#ifndef SIRIUS_IMGUI_THEME_HPP
#define SIRIUS_IMGUI_THEME_HPP

// The Modernist design tokens (docs/design/README.md) as the single source
// of colours, fonts and metrics for the application. Widgets read these
// constants when they draw; everything stock-ImGui is styled through the
// ImGuiStyle that applyTheme() installs. Flat: 0 px radius, 2 px rules between regions, 1 px
// between rows, shadows only on floating panels.
//
// Sizes: every metric here and in the panels is in design pixels, the ones
// docs/design counts in. px() turns them into pixels of this display (the
// monitor's content scale times the user's zoom), and the font helpers do
// the same, so a panel written against the 1600 x 960 design is right on a
// 150 % laptop as well.

#include <imgui.h>

#include <array>
#include <cstdint>
#include <string>

namespace sirius::app::gui::theme {

    // --- colours -----------------------------------------------------------
    constexpr ImU32 rgb(int r, int g, int b, int a = 255) {
        return (static_cast<ImU32>(a) << 24) | (static_cast<ImU32>(b) << 16) | (static_cast<ImU32>(g) << 8) |
               static_cast<ImU32>(r);
    }

    inline constexpr ImU32 kBg = rgb(0xf3, 0xf2, 0xf2);
    inline constexpr ImU32 kSurface = rgb(0xea, 0xe9, 0xe9);
    inline constexpr ImU32 kText = rgb(0x20, 0x1e, 0x1d);
    inline constexpr ImU32 kDivider = rgb(0xa6, 0xa5, 0xa4);          // text @ 40 % on bg
    inline constexpr ImU32 kAccent = rgb(0xec, 0x30, 0x13);
    inline constexpr ImU32 kAccent600 = rgb(0xdd, 0x2b, 0x0f);
    inline constexpr ImU32 kAccent700 = rgb(0xae, 0x18, 0x00);
    inline constexpr ImU32 kNeutral200 = rgb(0xea, 0xe7, 0xe7);
    inline constexpr ImU32 kNeutral300 = rgb(0xd7, 0xd3, 0xd3);
    inline constexpr ImU32 kNeutral400 = rgb(0xba, 0xb6, 0xb6);
    inline constexpr ImU32 kNeutral500 = rgb(0x9b, 0x97, 0x97);
    // The design's neutral-600 is #7d7979, which is 3.9:1 on the background:
    // below the 4.5:1 WCAG AA asks of the 10-12 px captions that use it.
    // Darkened to 5.0:1.
    inline constexpr ImU32 kNeutral600 = rgb(0x6b, 0x67, 0x67);
    // Text that has to stay the accent (11 px errors, links, the parameters
    // kicker): the same red, dark enough to pass. Fills and rules keep kAccent.
    inline constexpr ImU32 kAccentText = rgb(0xc6, 0x22, 0x00);
    inline constexpr ImU32 kNeutral700 = rgb(0x60, 0x5d, 0x5d);
    inline constexpr ImU32 kNeutral800 = rgb(0x44, 0x41, 0x41);
    inline constexpr ImU32 kNeutral900 = rgb(0x2d, 0x2b, 0x2b);
    inline constexpr ImU32 kViewerGround = rgb(0x0a, 0x09, 0x09);
    inline constexpr ImU32 kViewerText = rgb(0xf3, 0xf2, 0xf2);
    inline constexpr ImU32 kTransparent = rgb(0, 0, 0, 0);

    // `c` with its alpha multiplied by `opacity` (0..1).
    ImU32 withAlpha(ImU32 c, float opacity);
    // Linear blend of two colours, t = 0 -> a.
    ImU32 mix(ImU32 a, ImU32 b, float t);
    ImVec4 vec(ImU32 c);
    // A linear 0..1 (r, g, b) triple as the core keeps channel and label colours.
    ImU32 fromFloat(const std::array<float, 3>& c, float alpha = 1.0f);
    // "#ec3013"
    std::string hex(ImU32 c);

    // --- type ----------------------------------------------------------------
    constexpr float kBodyPx = 13;
    constexpr float kSmallPx = 11;
    constexpr float kCaptionPx = 10;       // uppercase, 0.1 em tracking
    constexpr float kH4Px = 20;
    constexpr float kH3Px = 24;
    constexpr float kBrandPx = 15;
    constexpr float kMonoPx = 15;

    enum class Weight { Regular,     // 400
                        SemiBold,    // 600
                        Bold,        // 700
                        ExtraBold }; // 800: headings

    // Archivo with the platform's fallbacks merged in for what it does not
    // carry (Greek, arrows, symbols); null until loadFonts() ran.
    ImFont* font(Weight w = Weight::Regular);
    ImFont* mono();
    // Archivo Regular with the caption's 0.1 em tracking.
    ImFont* captionFont();

    // RAII: text drawn while it lives uses this face at `designPx`.
    //     { theme::FontScope f(20, theme::Weight::ExtraBold); ImGui::TextUnformatted("Contrast"); }
    class FontScope {
    public:
        explicit FontScope(float designPx, Weight w = Weight::Regular);
        FontScope(float designPx, ImFont* face);
        ~FontScope();
        FontScope(const FontScope&) = delete;
        FontScope& operator=(const FontScope&) = delete;
    };
    // Width / size of `text` in that face, in display pixels.
    ImVec2 textSize(const char* text, float designPx, Weight w = Weight::Regular, const char* end = nullptr);
    ImVec2 textSize(const std::string& text, float designPx, Weight w = Weight::Regular);

    // --- metrics (design pixels) ---------------------------------------------
    constexpr float kTitleBarH = 38;
    constexpr float kViewerToolbarH = 40;
    constexpr float kStatusBarH = 26;
    constexpr float kOpsDockW = 290;
    constexpr float kParamsDockW = 320;
    constexpr float kAssistantW = 330;
    constexpr float kToolStripW = 36;
    constexpr float kDiagnosticsH = 250;
    constexpr float kDiagnosticsHeaderH = 34;
    constexpr float kRule = 2;
    constexpr float kHairline = 1;
    constexpr float kInputH = 32;          // text fields, spin boxes, dropdowns
    constexpr float kBorder = 1.5f;        // control borders

    // Display pixels per design pixel.
    float scale();
    void setScale(float s);
    inline float px(float designPx) { return designPx * scale(); }
    inline ImVec2 px(float x, float y) { return ImVec2(x * scale(), y * scale()); }
    // A stroke width that lands on whole display pixels (1.5 -> 1 or 2, never
    // two grey lines), and a coordinate snapped to the pixel grid.
    float crispPen(float designPx);
    float snap(float v);

    // Loads the bundled Archivo faces (beside the executable, an installed
    // tree, the source tree) with the platform fallbacks; false when Archivo
    // was not found and the default face stands in.
    bool loadFonts();
    // Installs the ImGui and ImPlot styles generated from the tokens.
    void applyTheme();

} // namespace sirius::app::gui::theme

#endif // SIRIUS_IMGUI_THEME_HPP
