// Tests of the help-page module (app/core/help_pages): every shipped page
// parses with the fields the help window shows, the LaTeX subset renders to
// the expected HTML and Markdown tables become HTML tables.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/app_paths.hpp"
#include "core/help_pages.hpp"

using namespace sirius::app;
namespace fs = std::filesystem;

namespace {

    bool contains(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }

    const char* kSample = R"(---
title: Sample operation
figure: A figure caption
figure_path: sample.png
---

First paragraph is the intro with $x^2$ math.
It continues on a second line.

$$
\tilde{S}(\mathbf{k}) = \frac{a}{b}\,A(\mathbf{k})
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Wiener** <br> 10⁻⁴ – 10⁻² | Regularisation $w$; larger blurs. $\tilde{S} = \frac{1}{w^2}$ |
| **Plain** | No range, no formula, has a pipe in math $|a|$ though. |

## Note

Footer note here.
)";
} // namespace

TEST_CASE("parseHelpMarkdown reads front matter, intro, formula, parameters and note", "[app][help]") {
    const HelpPage page = parseHelpMarkdown("sample", kSample);
    CHECK(page.kind == "sample");
    CHECK(page.title == "Sample operation");
    CHECK(page.figure == "A figure caption");
    CHECK(page.figurePath == "sample.png");
    CHECK(page.intro == "First paragraph is the intro with $x^2$ math. It continues on a second line.");
    CHECK(page.tex == R"(\tilde{S}(\mathbf{k}) = \frac{a}{b}\,A(\mathbf{k}))");
    REQUIRE(page.params.size() == 2);
    CHECK(page.params[0].name == "Wiener");
    CHECK(page.params[0].range == "10⁻⁴ – 10⁻²");
    CHECK(page.params[0].body == "Regularisation $w$; larger blurs.");
    CHECK(page.params[0].tex == R"(\tilde{S} = \frac{1}{w^2})");
    CHECK(page.params[1].name == "Plain");
    CHECK(page.params[1].range.empty());
    CHECK(page.params[1].tex.empty());
    CHECK(contains(page.params[1].body, "$|a|$"));
    CHECK(page.note == "Footer note here.");
    CHECK(page.markdown == kSample);
}

TEST_CASE("every shipped help page parses with the fields the window shows", "[app][help]") {
    const fs::path dir = helpDirectory();
    REQUIRE(fs::is_directory(dir));
    int pages = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".md") continue;
        ++pages;
        const std::string kind = entry.path().stem().string();
        const HelpPage page = loadHelpPage(kind);
        INFO("page " << kind);
        CHECK(page.path == entry.path().string());
        CHECK_FALSE(page.title.empty());
        CHECK(page.title != kind);
        CHECK_FALSE(page.intro.empty());
        CHECK_FALSE(page.tex.empty());
        CHECK_FALSE(page.figure.empty());
        CHECK_FALSE(page.params.empty());
        for (const HelpParam& p : page.params) {
            CHECK_FALSE(p.name.empty());
            CHECK_FALSE(p.body.empty());
            // the formulas the pages carry must render without leaking raw commands
            const std::string html = latexToHtml(p.tex, false);
            CHECK_FALSE(contains(html, "\\frac"));
        }
        const std::string display = latexToHtml(page.tex, true);
        CHECK_FALSE(display.empty());
        CHECK_FALSE(contains(display, "\\"));
        // the whole page renders
        const std::string html = helpMarkdownToHtml(page.markdown, dir.string());
        CHECK(contains(html, "<table"));
    }
    CHECK(pages >= 20);
    // operations the app ships
    for (const char* kind : {"load", "sim", "decon", "volrec", "einsum", "maxproj", "meant", "contrast", "flatfield",
                             "bleach", "deskew", "croppad", "resample", "merge", "stitch", "register", "seg",
                             "threshold", "cleanup", "manual", "shortcuts", "plugin-api"})
        CHECK(fs::exists(dir / (std::string(kind) + ".md")));
}

TEST_CASE("loadHelpPage of an unknown kind yields the placeholder page", "[app][help]") {
    const HelpPage page = loadHelpPage("no-such-operation");
    CHECK(page.kind == "no-such-operation");
    CHECK(contains(page.intro, "No help page yet"));
    CHECK(page.params.empty());
    CHECK(contains(page.path, "no-such-operation.md"));
}

TEST_CASE("latexToHtml renders fractions, scripts, Greek and bold", "[app][help][latex]") {
    SECTION("inline fraction uses a fraction slash with sup/sub") {
        const std::string html = latexToHtml(R"(\frac{a}{b})", false);
        CHECK(html == "<sup><i>a</i></sup>\xE2\x81\x84<sub><i>b</i></sub>");
    }
    SECTION("display fraction becomes a stacked table") {
        const std::string html = latexToHtml(R"(S = \frac{a}{b} A)", true);
        CHECK(contains(html, "<table"));
        CHECK(contains(html, "border-bottom"));
        CHECK(contains(html, "<i>a</i>"));
        CHECK(contains(html, "<i>b</i>"));
        CHECK(contains(html, "<i>S</i> = "));
    }
    SECTION("display without fractions stays a single span") {
        const std::string html = latexToHtml(R"(a + b)", true);
        CHECK_FALSE(contains(html, "<table"));
        CHECK(contains(html, "<i>a</i> + <i>b</i>"));
    }
    SECTION("subscripts and superscripts, braces and chains") {
        CHECK(latexToHtml("x^2", false) == "<i>x</i><sup>2</sup>");
        CHECK(latexToHtml("k_{0}", false) == "<i>k</i><sub>0</sub>");
        CHECK(latexToHtml(R"(\tilde{O}^*_m)", false) == "<i>O\xCC\x83</i><sup>∗</sup><sub><i>m</i></sub>");
        CHECK(latexToHtml("p_{lo}", false) == "<i>p</i><sub><i>l</i><i>o</i></sub>");
    }
    SECTION("Greek letters, operators and symbols") {
        CHECK(contains(latexToHtml(R"(\lambda \|\nabla o\|_1)", false), "λ"));
        CHECK(contains(latexToHtml(R"(\lambda \|\nabla o\|_1)", false), "∇"));
        CHECK(contains(latexToHtml(R"(\lambda \|\nabla o\|_1)", false), "‖"));
        CHECK(contains(latexToHtml(R"(a \cdot b \times c)", false), "·"));
        CHECK(contains(latexToHtml(R"(a \cdot b \times c)", false), "×"));
        CHECK(contains(latexToHtml(R"(x \in \mathbb{N})", false), "∈"));
        CHECK(contains(latexToHtml(R"(x \in \mathbb{N})", false), "ℕ"));
        CHECK(contains(latexToHtml(R"(A \rightarrow B)", false), "→"));
        CHECK(contains(latexToHtml(R"(\sum_{m} x_m)", false), "∑<sub><i>m</i></sub>"));
        CHECK(contains(latexToHtml(R"(\prod_{j<i}(1-\alpha_j))", false), "∏"));
        CHECK(contains(latexToHtml(R"(i \mid o)", false), "∣"));
        CHECK(contains(latexToHtml(R"(o \ast h^{\star})", false), "∗"));
        CHECK(contains(latexToHtml(R"(o \ast h^{\star})", false), "⋆"));
    }
    SECTION("\\mathbf is upright bold, \\text is upright, \\hat adds a combining mark") {
        CHECK(latexToHtml(R"(\mathbf{k})", false) == "<b style=\"font-style: normal\">k</b>");
        const std::string t = latexToHtml(R"(\text{uint16})", false);
        CHECK(contains(t, "uint16"));
        CHECK_FALSE(contains(t, "<i>"));
        CHECK(contains(latexToHtml(R"(\hat{o}^{(k+1)})", false), "o\xCC\x82"));
        CHECK(contains(latexToHtml(R"(\hat{o}^{(k+1)})", false), "<sup>(<i>k</i> + 1)</sup>"));
    }
    SECTION("\\left \\right delimiters and spacing commands") {
        const std::string html = latexToHtml(R"(\left[ \frac{i}{o} \right]\,x\quad y)", false);
        CHECK(contains(html, "["));
        CHECK(contains(html, "]"));
        CHECK(contains(html, "\xE2\x80\x89"));   // thin space
        CHECK(contains(html, "\xE2\x80\x83"));   // em space
        CHECK_FALSE(contains(html, "left"));
        CHECK_FALSE(contains(html, "right"));
    }
    SECTION("cases environment") {
        const std::string inl = latexToHtml(R"(\begin{cases} a & \text{inside} \\ f & \text{otherwise} \end{cases})", false);
        CHECK(contains(inl, "inside"));
        CHECK(contains(inl, "otherwise"));
        CHECK_FALSE(contains(inl, "begin"));
        const std::string disp = latexToHtml(R"(O = \begin{cases} a & \text{inside} \\ f & \text{otherwise} \end{cases})", true);
        CHECK(contains(disp, "rowspan"));
        CHECK(contains(disp, "inside"));
    }
    SECTION("unknown commands keep their name, empty input is empty") {
        CHECK(contains(latexToHtml(R"(\foo x)", false), "foo"));
        CHECK(latexToHtml("", false).empty());
        CHECK(latexToHtml("   ", true).empty());
    }
    SECTION("HTML in the source is escaped") {
        CHECK(contains(latexToHtml("a < b", false), "&lt;"));
    }
}

TEST_CASE("helpMarkdownToHtml renders tables, headings, lists, inline styles and math", "[app][help][markdown]") {
    const std::string md = R"(---
title: T
---

# Heading one

Intro with **bold**, *italic*, `code`, a [link](http://x) and $x^2$.

$$
\frac{a}{b}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Name** <br> 0 – 1 | Body text $\alpha$ |

- item one
- item two

1. first
2. second

![fig](figure.png)
)";
    const std::string html = helpMarkdownToHtml(md, "/base");
    CHECK_FALSE(contains(html, "title: T"));
    CHECK(contains(html, "font-size: 20px; font-weight: 800"));
    CHECK(contains(html, "Heading one"));
    CHECK(contains(html, "<b>bold</b>"));
    CHECK(contains(html, "<i>italic</i>"));
    CHECK(contains(html, "<code"));
    CHECK(contains(html, "href=\"http://x\""));
    CHECK(contains(html, "<i>x</i><sup>2</sup>"));
    CHECK(contains(html, "border-bottom"));          // display fraction
    CHECK(contains(html, "<table"));
    CHECK(contains(html, "<td"));
    CHECK(contains(html, "<b>Name</b>"));
    CHECK(contains(html, "0 – 1"));
    CHECK(contains(html, "α"));
    CHECK(contains(html, "<ul"));
    CHECK(contains(html, "<li>item two</li>"));
    CHECK(contains(html, "<ol"));
    CHECK(contains(html, "<li>second</li>"));
    CHECK(contains(html, "src=\"/base/figure.png\""));
    CHECK_FALSE(contains(html, "|---|"));
}

TEST_CASE("an image relative to the page resolves with '/' separators on every platform", "[app][help]") {
    // The rendered attribute is a URL, so the separator is '/' even where the
    // native one is '\'; a native join emitted "/base\figure.png" on Windows.
    const std::string html = helpMarkdownToHtml("![fig](figure.png)", "/base");
    CHECK(contains(html, "src=\"/base/figure.png\""));
    CHECK_FALSE(contains(html, "\\"));
    CHECK(contains(helpMarkdownToHtml("![fig](sub/figure.png)", "/base"), "src=\"/base/sub/figure.png\""));
    // a real page directory, whichever way this platform spells one
    const std::string real =
        helpMarkdownToHtml("![fig](figure.png)", (fs::temp_directory_path() / "sirius-help").string());
    CHECK_FALSE(contains(real, "\\"));
    CHECK(contains(real, "sirius-help/figure.png"));
    // a URL target is passed through untouched
    CHECK(contains(helpMarkdownToHtml("![f](http://x/y.png)", "/base"), "src=\"http://x/y.png\""));
    // an empty base directory leaves the target relative
    CHECK(contains(helpMarkdownToHtml("![f](y.png)", ""), "src=\"y.png\""));
}

TEST_CASE("helpDirectory honours SIRIUS_HELP_DIR and the hint", "[app][help]") {
    const fs::path tmp = fs::temp_directory_path() / "sirius-help-test-dir";
    fs::create_directories(tmp);
    std::ofstream(tmp / "custom.md") << "---\ntitle: Custom\n---\n\nIntro.\n\n$$ a $$\n";
    CHECK(helpDirectory(tmp.string()) == tmp.string());
    const HelpPage page = loadHelpPage("custom", tmp.string());
    CHECK(page.title == "Custom");
    CHECK(page.intro == "Intro.");
    CHECK(page.tex == "a");
    // a missing hint falls back to the shipped pages
    CHECK(fs::is_directory(helpDirectory("/definitely/not/a/dir")));
    fs::remove_all(tmp);
}

TEST_CASE("an installed tree's help pages come before the checkout's, the build tree's copy after", "[app][help]") {
    // The layout cmake --install produces: <prefix>/bin/sirius-app and
    // <prefix>/<datadir>/help. The checkout (SIRIUS_APP_SOURCE_DIR) exists
    // wherever this test runs, which is exactly the case to get right: an
    // install on the machine it was built on must read its own pages.
    const std::string relative = installedDataDirectoryFromBindir();
    REQUIRE(!relative.empty());
    const fs::path prefix = fs::temp_directory_path() / "sirius-installed-help-test";
    fs::remove_all(prefix);
    const fs::path bin = prefix / "bin";
    const fs::path installed = (bin / relative / "help").lexically_normal();
    fs::create_directories(bin);
    fs::create_directories(installed);
    struct Restore {
        ~Restore() { setApplicationDirectory({}); }
    } restore;
    setApplicationDirectory(bin.string());
    CHECK(installedDataDirectory("help") == installed.string());
    CHECK(fs::path(helpDirectory()) == installed);
    // an explicit hint and $SIRIUS_HELP_DIR still come first
    const fs::path hint = prefix / "hint";
    fs::create_directories(hint);
    CHECK(fs::path(helpDirectory(hint.string())) == hint);

    // not installed: the checkout, never the copy beside the executable, so
    // "Edit page" in a development build edits the file under version control
    fs::remove_all(prefix / "share");
    fs::create_directories(bin / "help");
    CHECK(installedDataDirectory("help").empty());
    CHECK(fs::path(besideApplication("help")) == (bin / "help").lexically_normal());
    CHECK(fs::path(helpDirectory()) != (bin / "help").lexically_normal());
    CHECK(fs::exists(fs::path(helpDirectory()) / "load.md"));

    // with no executable directory set (a test, a library user) nothing is found
    setApplicationDirectory({});
    CHECK(installedDataDirectory("help").empty());
    CHECK(besideApplication("help").empty());
    fs::remove_all(prefix);
}

TEST_CASE("Markdown math accepts the \\( \\) and \\[ \\] delimiters", "[app][help][math]") {
    using sirius::app::normalizeMathDelimiters;
    CHECK(normalizeMathDelimiters("a \\(x^2\\) b") == "a $x^2$ b");
    CHECK(normalizeMathDelimiters("\\[\\frac{a}{b}\\]") == "$$\\frac{a}{b}$$");
    // untouched inside code and inside existing math
    CHECK(normalizeMathDelimiters("`\\(x\\)`") == "`\\(x\\)`");
    CHECK(normalizeMathDelimiters("$a \\\\[2pt] b$") == "$a \\\\[2pt] b$");
    const std::string html = sirius::app::helpMarkdownToHtml("The window is \\(w = \\frac{1}{\\gamma}\\).", "");
    CHECK(html.find("γ") != std::string::npos);
    CHECK(html.find("\\(") == std::string::npos);
}
