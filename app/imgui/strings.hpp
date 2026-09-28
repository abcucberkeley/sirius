#ifndef SIRIUS_IMGUI_STRINGS_HPP
#define SIRIUS_IMGUI_STRINGS_HPP

// Small string helpers the panels share. Every string in the GUI layer is
// UTF-8 in a std::string, which is what Dear ImGui draws and what the core
// takes.

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::app::gui {

    // printf into a std::string.
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    inline std::string
    format(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        va_list copy;
        va_copy(copy, args);
        const int n = std::vsnprintf(nullptr, 0, fmt, copy);
        va_end(copy);
        std::string out;
        if (n > 0) {
            out.resize(static_cast<std::size_t>(n) + 1);
            std::vsnprintf(out.data(), out.size(), fmt, args);
            out.resize(static_cast<std::size_t>(n));
        }
        va_end(args);
        return out;
    }

    // Decodes one code point at `i` (advancing it); U+FFFD on a bad sequence.
    char32_t nextCodepoint(std::string_view s, std::size_t& i);
    void appendUtf8(std::string& out, char32_t c);

    // Caption style: the words in capitals, a unit or a symbol as written.
    //
    // Uppercasing the micro sign gives GREEK CAPITAL MU, which reads as an M:
    // a "µm / FRAME" column would say "MM / FRAME". A word holding a Greek
    // letter, the micro sign, a superscript or subscript, a degree or an
    // ångström sign is kept as written; every other word is uppercased
    // (ASCII and Latin-1 letters).
    std::string captionCase(std::string_view text);

    std::string toLower(std::string_view s);        // ASCII
    std::string toUpper(std::string_view s);        // ASCII
    std::string trimmed(std::string_view s);
    // Runs of whitespace (newlines included) become one space; trimmed.
    std::string simplified(std::string_view s);
    bool startsWith(std::string_view s, std::string_view prefix);
    bool endsWith(std::string_view s, std::string_view suffix);
    bool endsWithNoCase(std::string_view s, std::string_view suffix);
    bool containsNoCase(std::string_view s, std::string_view needle);
    std::vector<std::string> split(std::string_view s, char sep, bool skipEmpty = false);
    std::string join(const std::vector<std::string>& parts, std::string_view sep);
    std::string replaceAll(std::string s, std::string_view from, std::string_view to);

    // "12.8 GB", "412 MB", "48 kB" -- core's formatBytes, so every byte
    // readout in the application rounds the same way.
    std::string bytesText(std::uint64_t bytes);
    // "40 s", "3:05 min", "1 h 12 min"
    std::string durationText(double seconds);

    // File name helpers on UTF-8 paths (std::filesystem underneath). The
    // paths they make are written with '/', on Windows too, like the paths
    // the file dialogs return.
    std::string fileName(const std::string& path);            // "stack.ome.tif"
    std::string completeBaseName(const std::string& path);    // "stack.ome"
    std::string parentPath(const std::string& path);          // absolute
    std::string absolutePath(const std::string& path);
    bool pathExists(const std::string& path);
    bool isDirectory(const std::string& path);
    bool isFile(const std::string& path);

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_STRINGS_HPP
