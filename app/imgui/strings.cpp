#include "imgui/strings.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <system_error>

#include "core/ops/common.hpp"

namespace sirius::app::gui {

    char32_t nextCodepoint(std::string_view s, std::size_t& i) {
        const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
        const unsigned char b0 = byte(i);
        int extra = 0;
        char32_t c = 0;
        if (b0 < 0x80) {
            c = b0;
        } else if ((b0 & 0xE0) == 0xC0) {
            c = b0 & 0x1F;
            extra = 1;
        } else if ((b0 & 0xF0) == 0xE0) {
            c = b0 & 0x0F;
            extra = 2;
        } else if ((b0 & 0xF8) == 0xF0) {
            c = b0 & 0x07;
            extra = 3;
        } else {
            ++i;
            return 0xFFFD;
        }
        if (i + static_cast<std::size_t>(extra) >= s.size()) {   // truncated sequence
            if (extra > 0) {
                i = s.size();
                return 0xFFFD;
            }
        }
        for (int k = 1; k <= extra; ++k) {
            const unsigned char b = byte(i + static_cast<std::size_t>(k));
            if ((b & 0xC0) != 0x80) {
                ++i;
                return 0xFFFD;
            }
            c = (c << 6) | (b & 0x3F);
        }
        i += static_cast<std::size_t>(extra) + 1;
        return c;
    }

    void appendUtf8(std::string& out, char32_t c) {
        if (c < 0x80) {
            out += static_cast<char>(c);
        } else if (c < 0x800) {
            out += static_cast<char>(0xC0 | (c >> 6));
            out += static_cast<char>(0x80 | (c & 0x3F));
        } else if (c < 0x10000) {
            out += static_cast<char>(0xE0 | (c >> 12));
            out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (c & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (c >> 18));
            out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (c & 0x3F));
        }
    }

    namespace {
        bool isSpace(char32_t c) { return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == 0x00A0; }

        bool keepsCase(char32_t u) {
            return u == 0x00B5 || (u >= 0x0370 && u <= 0x03FF) || (u >= 0x2070 && u <= 0x209F) || u == 0x00B2 ||
                   u == 0x00B3 || u == 0x00B9 || u == 0x00B0 || u == 0x212B;
        }

        char32_t upper(char32_t c) {
            if (c >= U'a' && c <= U'z') return c - 32;
            // Latin-1 letters (à..þ but not the division sign), ÿ stays
            if (c >= 0x00E0 && c <= 0x00FE && c != 0x00F7) return c - 32;
            return c;
        }
    } // namespace

    std::string captionCase(std::string_view text) {
        std::string out;
        out.reserve(text.size());
        std::size_t i = 0;
        while (i < text.size()) {
            std::size_t j = i;
            const char32_t c = nextCodepoint(text, j);
            if (isSpace(c)) {
                out.append(text.substr(i, j - i));
                i = j;
                continue;
            }
            // one word
            std::size_t end = i;
            bool keep = false;
            std::vector<char32_t> word;
            while (end < text.size()) {
                std::size_t k = end;
                const char32_t w = nextCodepoint(text, k);
                if (isSpace(w)) break;
                keep = keep || keepsCase(w);
                word.push_back(w);
                end = k;
            }
            if (keep) out.append(text.substr(i, end - i));
            else
                for (char32_t w : word) appendUtf8(out, upper(w));
            i = end;
        }
        return out;
    }

    std::string toLower(std::string_view s) {
        std::string out(s);
        for (char& c : out)
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
        return out;
    }

    std::string toUpper(std::string_view s) {
        std::string out(s);
        for (char& c : out)
            if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
        return out;
    }

    std::string trimmed(std::string_view s) {
        std::size_t a = 0, b = s.size();
        while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
        return std::string(s.substr(a, b - a));
    }

    std::string simplified(std::string_view s) {
        std::string out;
        bool space = false;
        for (char ch : s) {
            if (std::isspace(static_cast<unsigned char>(ch))) {
                space = !out.empty();
                continue;
            }
            if (space) out += ' ';
            space = false;
            out += ch;
        }
        return out;
    }

    bool startsWith(std::string_view s, std::string_view prefix) {
        return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
    }

    bool endsWith(std::string_view s, std::string_view suffix) {
        return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
    }

    bool endsWithNoCase(std::string_view s, std::string_view suffix) {
        return endsWith(toLower(s), toLower(suffix));
    }

    bool containsNoCase(std::string_view s, std::string_view needle) {
        return toLower(s).find(toLower(needle)) != std::string::npos;
    }

    std::vector<std::string> split(std::string_view s, char sep, bool skipEmpty) {
        std::vector<std::string> out;
        std::size_t start = 0;
        while (start <= s.size()) {
            const std::size_t end = s.find(sep, start);
            const std::string_view part = s.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start);
            if (!part.empty() || !skipEmpty) out.emplace_back(part);
            if (end == std::string_view::npos) break;
            start = end + 1;
        }
        return out;
    }

    std::string join(const std::vector<std::string>& parts, std::string_view sep) {
        std::string out;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            if (i) out.append(sep);
            out += parts[i];
        }
        return out;
    }

    std::string replaceAll(std::string s, std::string_view from, std::string_view to) {
        if (from.empty()) return s;
        std::size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
        return s;
    }

    std::string bytesText(std::uint64_t bytes) { return formatBytes(bytes); }

    std::string durationText(double seconds) {
        if (seconds < 60.0) return format("%d s", static_cast<int>(seconds + 0.5));
        const int m = static_cast<int>(seconds / 60.0), sec = static_cast<int>(seconds + 0.5) % 60;
        if (m < 60) return format("%d:%02d min", m, sec);
        return format("%d h %d min", m / 60, m % 60);
    }

    namespace {
        std::filesystem::path fsPath(const std::string& p) { return std::filesystem::u8path(p); }
        std::string fromPath(const std::filesystem::path& p) { return p.u8string(); }
    } // namespace

    std::string fileName(const std::string& path) {
        std::filesystem::path p = fsPath(path);
        if (!p.has_filename() && p.has_parent_path()) p = p.parent_path();   // "dir/" -> "dir"
        return fromPath(p.filename());
    }

    std::string completeBaseName(const std::string& path) {
        const std::string name = fileName(path);
        const std::size_t dot = name.rfind('.');
        return dot == std::string::npos || dot == 0 ? name : name.substr(0, dot);
    }

    std::string absolutePath(const std::string& path) {
        std::error_code ec;
        const std::filesystem::path p = std::filesystem::absolute(fsPath(path), ec);
        return ec ? path : fromPath(p.lexically_normal());
    }

    std::string parentPath(const std::string& path) {
        std::filesystem::path p = fsPath(absolutePath(path));
        if (!p.has_filename() && p.has_parent_path()) p = p.parent_path();
        return fromPath(p.parent_path());
    }

    bool pathExists(const std::string& path) {
        std::error_code ec;
        return !path.empty() && std::filesystem::exists(fsPath(path), ec);
    }

    bool isDirectory(const std::string& path) {
        std::error_code ec;
        return !path.empty() && std::filesystem::is_directory(fsPath(path), ec);
    }

    bool isFile(const std::string& path) {
        std::error_code ec;
        return !path.empty() && std::filesystem::is_regular_file(fsPath(path), ec);
    }

} // namespace sirius::app::gui
