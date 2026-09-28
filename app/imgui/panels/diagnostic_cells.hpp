#ifndef SIRIUS_IMGUI_PANELS_DIAGNOSTIC_CELLS_HPP
#define SIRIUS_IMGUI_PANELS_DIAGNOSTIC_CELLS_HPP

// What the diagnostics dock is assembled from. None of it knows the
// workbench: the cells draw a Diagnostics struct (or one of its parts), so
// the same cells serve every operation kind. Every cell is a bg-coloured box
// with a 10 px uppercase caption (title left, meta right) on a 2 px
// divider-coloured grid. (app/qt/panels/diagnostic_cells.cpp)
//
// Immediate mode: a cell is a function that draws into a rectangle of the
// current window. The only state is in DiagnosticsBody, which keeps the
// Diagnostics it was given and the textures rendered from its images.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <imgui.h>

#include "core/diagnostics.hpp"
#include "imgui/gl.hpp"

namespace sirius::app::gui {

    // One byte per pixel, rows top to bottom.
    struct GrayImage {
        int width = 0, height = 0;
        std::vector<std::uint8_t> pixels;
        bool empty() const noexcept { return pixels.empty(); }
    };

    // Float plane -> 8-bit grayscale with a robust (percentile) window. A log
    // scale is applied by whoever fills the DiagnosticImage (spectrumImage
    // stores log10 of the power and sets DiagnosticImage::logScale to say so).
    GrayImage renderDiagnosticImage(const DiagnosticImage& image);

    namespace cells {

        struct Rect {
            ImVec2 min, max;
            float width() const noexcept { return max.x - min.x; }
            float height() const noexcept { return max.y - min.y; }
            // Shrunk by design pixels on each side.
            Rect inset(float left, float top, float right, float bottom) const;
        };

        // A font size in design pixels as Dear ImGui's draw list wants it.
        float fontPx(float designPx);
        // `text` cut to `width` display pixels in `face`, with an ellipsis.
        std::string elideIn(ImFont* face, float designPx, const std::string& text, float width);

        // The cell: a child window over (min, max) with the caption row
        // drawn. Returns the rectangle below the caption. endCell() must be
        // called whatever happened in between.
        Rect beginCell(const char* id, ImVec2 min, ImVec2 max, const std::string& title, const std::string& meta);
        // The same box without a caption (the label table, the cleanup tools).
        Rect beginBox(const char* id, ImVec2 min, ImVec2 max);
        void endCell();
        // The 10 px caption as the cells write it, at the cursor.
        float captionRowHeight();

        // One image on a black ground, aspect preserved, marks in accent.
        // `texture` null (or not valid): the ground and the placeholder.
        void image(const Rect& r, Texture* texture, const std::vector<DiagnosticMark>& marks, const std::string& placeholder);
        // Polyline over a baseline, optional dashed stop line, three labels.
        void curve(const char* id, const Rect& r, const DiagnosticCurve* curve);
        // Bars flush to the bottom; bins outside [lo, hi] in neutral-400.
        void histogram(const char* id, const Rect& r, const DiagnosticHistogram* histogram);
        // Key / value rows separated by hairlines, an optional lead paragraph
        // (the step summary) and a trailing muted line.
        void facts(const char* id, const Rect& r, const std::vector<DiagnosticFact>& facts, const std::string& lead = {},
                   const std::string& trailer = {});
        // Tile grid of an AlignmentInfo, highlighted tile accent-filled.
        void tileMap(const Rect& r, const AlignmentInfo& info);
        // A DiagnosticTable (accent cells in accent 800).
        void table(const char* id, const Rect& r, const DiagnosticTable& table);

        // --- dense tables (the diagnostics table, the labels, the tracks) ----
        // 11 px tabular text in 22 px rows, 10 px headers over a 2 px rule.
        // Push before BeginTable, pop after EndTable.
        void pushTableStyle();
        void popTableStyle();
        // The header row from the columns set up (click to sort when the
        // table is sortable); `tips` may be null.
        void tableHeaders(const char* const* tips = nullptr);
        // The 2 px rule under the header of the table that just ended.
        void tableHeaderRule(ImVec2 tableMin, float tableWidth, float headerHeight);
        // Height of a header row / of a body row, display pixels.
        float tableHeaderHeight();
        float tableRowHeight();

        // A tool tip for the last item that also shows when it is disabled.
        void tooltipAlways(const std::string& text);

    } // namespace cells

    // The per-kind grid built from a Diagnostics value: which cells, in
    // which columns, for the active tab. Placeholders describe what a run
    // would fill in when the diagnostics are empty.
    class DiagnosticsBody {
    public:
        struct Context {
            std::string stepSummary;      // one line about the selected step
            std::string inputShape;       // "c2 t40 z48 y2048 x2048"
            std::string outputShape;
            std::string estimate;         // "≈ 412 MB output · cache memory"
        };

        // Replaces what is shown; the textures are rendered again when they
        // are next drawn.
        void setDiagnostics(Diagnostics d, DiagnosticsKind kind, Context ctx);
        // Fills the rest of the current window.
        void draw(int tab);
        // Tab names for a kind (Diagnostics::tabs when present).
        static std::vector<std::string> tabNames(const Diagnostics& d, DiagnosticsKind kind);

    private:
        struct Cell {
            std::string title, meta;
            int stretch = 1;
            float fixedWidth = 0.0f;      // design pixels, 0 = share the rest
            std::function<void(const cells::Rect&)> content;
        };
        Texture* texture(std::size_t index);
        void imageCell(std::vector<Cell>& out, std::size_t index, const std::string& fallbackTitle, const std::string& fallbackMeta,
                       const std::string& placeholder, float fixedWidth = 0.0f);

        Diagnostics d_;
        DiagnosticsKind kind_ = DiagnosticsKind::Generic;
        Context ctx_;
        std::vector<Texture> textures_;
        std::vector<char> rendered_;      // per image: the texture is up to date (or the image is empty)
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_PANELS_DIAGNOSTIC_CELLS_HPP
