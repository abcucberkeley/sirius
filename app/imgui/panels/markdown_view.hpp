#ifndef SIRIUS_IMGUI_PANELS_MARKDOWN_VIEW_HPP
#define SIRIUS_IMGUI_PANELS_MARKDOWN_VIEW_HPP

// Rich text for the help pages: Markdown with $...$ LaTeX, laid out with
// Dear ImGui's fonts and drawn into an ImDrawList.
//
// Three steps, kept apart so that a page is parsed and measured once and
// then only drawn:
//
//     text  ->  boxes        words, spaces, sub- and superscripts, stacked
//                            fractions; every box knows its width and how
//                            far it reaches above and below the baseline
//     boxes ->  Layout       lines broken at the spaces for a given width;
//                            the result is a flat list of positioned texts,
//                            rectangles, accents, images and link areas
//     Layout -> screen       draw() at an origin, every frame
//
// A Layout is built for one width and one display scale; whoever keeps it
// rebuilds it when either changes.
//
// The maths is the subset core/help_pages.hpp renders (latexToHtml): that
// function is the parser, and its small, regular HTML (<i>, <b>, <sub>,
// <sup>, a two-row table per fraction) is what is laid out here, so both
// applications understand exactly the same commands.

#include <imgui.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "imgui/gl.hpp"
#include "imgui/theme.hpp"

namespace sirius::app::gui::markdown {

    enum class Face { Body,       // Archivo
                      Mono,       // inline code, code blocks
                      Caption };  // Archivo with the caption's tracking; the text is put in caption case

    struct TextStyle {
        float px = theme::kBodyPx;                        // design pixels
        theme::Weight weight = theme::Weight::Regular;
        ImU32 color = theme::kText;
        bool italic = false;
        Face face = Face::Body;
        // Distance between the baselines of a paragraph, in units of `px`.
        float leading = 1.38f;
    };

    enum class Align { Left,
                       Center };

    // The textures of the images a page shows, by path. Created and destroyed
    // on the GUI thread (when a layout is built, and with the owner).
    class Images {
    public:
        // Null when the file cannot be read as an image (SVG, PDF, missing).
        std::shared_ptr<Texture> get(const std::string& path);
        void clear();

    private:
        std::map<std::string, std::shared_ptr<Texture>> textures_;
    };

    // Where relative image paths start and where the textures are kept.
    struct Context {
        std::string baseDir;
        Images* images = nullptr;
    };

    // Positioned primitives, relative to the layout's top left corner, in
    // display pixels.
    class Layout {
    public:
        float width = 0.0f;
        float height = 0.0f;

        bool empty() const noexcept { return texts_.empty() && rects_.empty() && images_.empty() && accents_.empty(); }

        // --- building ----------------------------------------------------------
        void fill(ImVec2 min, ImVec2 max, ImU32 color);
        // An outline inside (min, max), `designPx` wide; dashed for the drop zone.
        void frame(ImVec2 min, ImVec2 max, ImU32 color, float designPx = 1.0f, bool dashed = false);
        void image(ImVec2 min, ImVec2 max, std::shared_ptr<Texture> texture);
        void link(ImVec2 min, ImVec2 max, const std::string& url);
        // Another layout, its corner at (x, y) of this one. Grows this one to hold it.
        void place(const Layout& part, float x, float y);

        // --- showing -----------------------------------------------------------
        void draw(ImDrawList* dl, ImVec2 origin) const;
        // The target of the link under `mouse` (display pixels), or null.
        const std::string* linkAt(ImVec2 origin, ImVec2 mouse) const;

        // Used by the boxes while a layout is built.
        struct Text {
            ImVec2 pos;              // top left of the line box of the font
            float end = 0.0f;        // x where the text ends
            float baseline = 0.0f;
            ImFont* font = nullptr;
            float size = 0.0f;       // display pixels
            ImU32 color = 0;
            bool italic = false;
            std::string text;
        };
        enum class Accent { Tilde,
                            Hat,
                            Bar,
                            Vec,
                            Dot,
                            DDot };
        struct AccentMark {
            Accent kind = Accent::Tilde;
            ImVec2 centre;           // of the mark
            float width = 0.0f;      // of the glyph below it
            float size = 0.0f;       // font size, display pixels
            ImU32 color = 0;
        };
        void text(const Text& t);
        void accent(const AccentMark& a);

    private:
        struct Rect {
            ImVec2 min, max;
            ImU32 color = 0;
            float pen = 0.0f;        // design px; 0 = filled
            bool dashed = false;
        };
        struct Picture {
            ImVec2 min, max;
            std::shared_ptr<Texture> texture;
        };
        struct Link {
            ImVec2 min, max;
            std::string url;
        };
        std::vector<Rect> rects_;
        std::vector<Text> texts_;
        std::vector<AccentMark> accents_;
        std::vector<Picture> images_;
        std::vector<Link> links_;
    };

    // One paragraph of inline Markdown (bold, italic, code, $maths$, links,
    // images, <br>), wrapped at `width` display pixels.
    Layout paragraph(const std::string& markdown, const TextStyle& style, float width, const Context& context = {},
                     Align align = Align::Left);
    // The same without any markup: the text as it is.
    Layout plain(const std::string& text, const TextStyle& style, float width, Align align = Align::Left);
    // A formula. Display: fractions are stacked and the formula is scaled
    // down when it is wider than `width`; inline: wrapped like a paragraph.
    Layout formula(const std::string& tex, bool display, const TextStyle& style, float width);
    // A whole document: headings, paragraphs, bullet and numbered lists,
    // tables, fenced code, $$ display formulas $$, images.
    Layout blocks(const std::string& markdown, const TextStyle& body, float width, const Context& context = {});

    // Submits the layout as an item at the cursor and draws it; a click on a
    // link opens it in the browser. Returns true when a link was opened.
    bool show(const Layout& layout);

} // namespace sirius::app::gui::markdown

#endif // SIRIUS_IMGUI_PANELS_MARKDOWN_VIEW_HPP
