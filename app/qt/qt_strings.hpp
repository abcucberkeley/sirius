#ifndef SIRIUS_APP_QT_STRINGS_HPP
#define SIRIUS_APP_QT_STRINGS_HPP

// QString <-> std::string without QString::toStdString/fromStdString.
//
// Those are inline in the Qt headers, but on MSVC a dllimport class may have
// its inline members taken from the DLL instead (Debug builds do), and the
// DLL's copy was compiled against a release std::string, whose layout differs
// from the debug one. A Debug sirius-app against a release-only Qt (the usual
// case: prebuilt Qt kits ship release DLLs) then corrupts every path it passes
// to the library. Going through QByteArray keeps all std::string code on our
// side of the boundary, for every Qt build.

#include <cstddef>
#include <string>

#include <QByteArray>
#include <QString>

namespace sirius::app {

    inline std::string toStd(const QString& s) {
        const QByteArray utf8 = s.toUtf8();
        return std::string(utf8.constData(), static_cast<std::size_t>(utf8.size()));
    }

    inline QString fromStd(const std::string& s) {
        return QString::fromUtf8(s.data(), static_cast<int>(s.size()));
    }

    // Caption style: the words in capitals, a unit or a symbol as written.
    //
    // QString::toUpper -- and QFont::AllUppercase and a style sheet's
    // text-transform, which do the same while drawing -- turns the micro sign
    // into GREEK CAPITAL MU, which reads as an M: a "µm / FRAME" column said
    // "MM / FRAME", a millimetre, and k₀, λ and σ lost their case with it. A
    // word holding a Greek letter, the micro sign, a superscript or subscript,
    // a degree or an ångström sign is kept as written; every other word is
    // uppercased. So captions are cased here, never by a font or a style sheet.
    inline QString captionCase(const QString& text) {
        auto symbol = [](QChar c) {
            const char16_t u = c.unicode();
            return u == 0x00B5 || (u >= 0x0370 && u <= 0x03FF) || (u >= 0x2070 && u <= 0x209F) || u == 0x00B2 ||
                   u == 0x00B3 || u == 0x00B9 || u == 0x00B0 || u == 0x212B;
        };
        QString out;
        out.reserve(text.size());
        qsizetype i = 0;
        while (i < text.size()) {
            if (text.at(i).isSpace()) {
                out += text.at(i++);
                continue;
            }
            qsizetype end = i;
            bool keep = false;
            for (; end < text.size() && !text.at(end).isSpace(); ++end) keep = keep || symbol(text.at(end));
            const QString word = text.mid(i, end - i);
            out += keep ? word : word.toUpper();
            i = end;
        }
        return out;
    }

} // namespace sirius::app

#endif // SIRIUS_APP_QT_STRINGS_HPP
