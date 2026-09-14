#ifndef SIRIUS_APP_HELP_PAGES_HPP
#define SIRIUS_APP_HELP_PAGES_HPP

// Help pages are Markdown files with $...$ / $$...$$ LaTeX and images,
// one per operation, stored in app/help next to the operation code and
// installed beside the executable so users can edit them. The core locates
// and parses them; the Qt layer renders the HTML.

#include <string>
#include <vector>

namespace sirius::app {

    struct HelpParam {
        std::string name, range, body, tex;
    };

    struct HelpPage {
        std::string kind;
        std::string title;
        std::string intro;
        std::string tex;                  // display formula
        std::string figure;               // caption of the figure slot
        std::string figurePath;           // image beside the page, if any
        std::vector<HelpParam> params;
        std::string note;
        std::string markdown;             // the whole source
        std::string path;                 // file it came from ("" = built in)
    };

    // Directory searched for "<kind>.md": $SIRIUS_HELP_DIR, then `hint`, then
    // an installed tree's share/sirius/help, the source tree's app/help, and
    // the copy beside the executable (core/app_paths.hpp).
    std::string helpDirectory(const std::string& hint = {});
    HelpPage loadHelpPage(const std::string& kind, const std::string& hint = {});
    // A page that lives in memory (a plugin's docstring); a file of the same
    // kind in the help directory still wins so users can override it.
    void registerHelpPage(const std::string& kind, const std::string& markdown);
    HelpPage parseHelpMarkdown(const std::string& kind, const std::string& markdown);

    // LaTeX (the subset the pages use: fractions, sub/superscripts, Greek,
    // \sum, \prod, \mathbf, \tilde, \hat, \text, \left..\right, \cdot,
    // \times, \in, \mid, \ast, \star, \nabla, \rightarrow) to HTML with
    // Unicode, <sub>, <sup> and a two-row table for fractions. Unknown
    // commands keep their name.
    std::string latexToHtml(const std::string& tex, bool display);
    // Markdown with $..$ math to HTML (headings, paragraphs, lists, bold,
    // italic, code, tables, images), for QTextBrowser. Also accepts the
    // \(..\) and \[..\] delimiters language models favour.
    std::string helpMarkdownToHtml(const std::string& markdown, const std::string& baseDir);
    // \(..\) -> $..$ and \[..\] -> $$..$$ outside code and existing math.
    std::string normalizeMathDelimiters(const std::string& markdown);

} // namespace sirius::app

#endif // SIRIUS_APP_HELP_PAGES_HPP
