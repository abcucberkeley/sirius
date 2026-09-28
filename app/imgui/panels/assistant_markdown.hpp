#ifndef SIRIUS_IMGUI_ASSISTANT_MARKDOWN_HPP
#define SIRIUS_IMGUI_ASSISTANT_MARKDOWN_HPP

// The assistant's answers are Markdown with $..$ math. The subset that
// core/help_pages turns into HTML (helpMarkdownToHtml) -- headings,
// paragraphs, lists, bold, italic, code, links, tables, inline and display
// math -- is parsed here into blocks of styled spans, broken into lines for
// a width and drawn into the draw list. Fenced
// code blocks, which language models write and the help pages do not, are
// set in the monospace face on the surface colour.
//
// A Layout is made for one width at one display scale; the panel keeps it
// with the message and asks for a new one when either changed.

#include <string>
#include <vector>

#include <imgui.h>

namespace sirius::app::gui::assistant_markdown {

    struct Span {
        std::string text;
        bool bold = false;
        bool italic = false;       // no italic face is bundled: drawn upright
        bool code = false;         // monospace on the surface colour
        int script = 0;            // +1 superscript, -1 subscript (math)
        bool lineBreak = false;    // <br>
        std::string href;          // a link's target
    };

    struct Block {
        enum class Kind { Paragraph,
                          Heading,
                          Bullet,
                          Numbered,
                          Code,
                          Math,
                          TableHeader,
                          TableRow,
                          Rule };
        Kind kind = Kind::Paragraph;
        int level = 0;             // heading level, list nesting
        std::string marker;        // "1." of a numbered item
        std::vector<Span> spans;
    };

    struct Document {
        std::vector<Block> blocks;
    };

    Document parse(const std::string& markdown);
    // Text as typed: line breaks kept, nothing interpreted (the user's bubble).
    Document plain(const std::string& text);

    struct Run {
        ImVec2 pos;                // relative to the layout's origin, display pixels
        ImVec2 size;
        std::string text;
        float px = 13;             // design pixels
        int face = 0;              // 0 regular, 1 bold, 2 extra bold, 3 monospace
        ImU32 color = 0;
        bool fill = false;         // inline code: the surface colour behind it
        int link = -1;             // index into Layout::links
    };

    struct Fill {
        ImVec2 min, max;
        ImU32 color = 0;
        bool outline = false;
    };

    struct Layout {
        float width = -1.0f;       // what it was made for
        float scale = 0.0f;
        ImVec2 size;               // what it takes (size.x <= width)
        std::vector<Fill> fills;
        std::vector<Run> runs;
        std::vector<std::string> links;
    };

    // Breaks `doc` into lines no wider than `width` display pixels, in
    // `bodyPx` text of `color`. Inside a frame (it measures with the fonts).
    Layout layout(const Document& doc, float width, float bodyPx, ImU32 color);

    // Draws at `origin`; the index of the link under the mouse, or -1.
    int draw(ImDrawList* dl, ImVec2 origin, const Layout& layout);

} // namespace sirius::app::gui::assistant_markdown

#endif // SIRIUS_IMGUI_ASSISTANT_MARKDOWN_HPP
