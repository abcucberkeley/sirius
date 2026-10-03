#include "sirius/tiff_metadata.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <sstream>

namespace sirius {

    namespace {

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        std::string trim(const std::string& s) {
            const auto a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
            return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
        }

        double timeUnitToS(const std::string& unitIn) {
            const std::string u = lower(trim(unitIn));
            if (u.empty() || u == "s" || u == "sec" || u == "second" || u == "seconds") return 1.0;
            if (u == "ms") return 1e-3;
            if (u == "us" || u == "\xc2\xb5s" || u == "\xce\xbcs") return 1e-6;
            if (u == "ns") return 1e-9;
            if (u == "min") return 60.0;
            if (u == "h") return 3600.0;
            return 1.0;
        }

        // A number as std::stod reads it (leading blanks, then the longest
        // prefix that parses), or `def` when there is none.
        double toDouble(const std::string& s, double def) {
            try {
                return std::stod(s);
            } catch (...) {
                return def;
            }
        }

        std::uint64_t toCount(double v) {
            if (!(v > 0.0)) return 0;
            if (v >= static_cast<double>(std::numeric_limits<std::uint32_t>::max())) return std::numeric_limits<std::uint32_t>::max();
            return static_cast<std::uint64_t>(v);
        }

        // The handful of XML entities OME-XML attribute values use, and
        // numeric references (µ appears as &#181; in some writers).
        std::string xmlUnescape(std::string s) {
            if (s.find('&') == std::string::npos) return s;
            struct E {
                const char* from;
                const char* to;
            };
            static const E ents[] = {{"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}};
            for (const E& e : ents) {
                std::size_t pos = 0;
                while ((pos = s.find(e.from, pos)) != std::string::npos) {
                    s.replace(pos, std::strlen(e.from), e.to);
                    pos += std::strlen(e.to);
                }
            }
            std::size_t pos = 0;
            while ((pos = s.find("&#", pos)) != std::string::npos) {
                const std::size_t end = s.find(';', pos);
                if (end == std::string::npos) break;
                const std::string num = s.substr(pos + 2, end - pos - 2);
                unsigned long code = 0;
                try {
                    code = num.size() > 1 && (num[0] == 'x' || num[0] == 'X') ? std::stoul(num.substr(1), nullptr, 16)
                                                                              : std::stoul(num);
                } catch (...) {
                    pos = end + 1;
                    continue;
                }
                std::string utf8;
                if (code < 0x80) {
                    utf8 += static_cast<char>(code);
                } else if (code < 0x800) {
                    utf8 += static_cast<char>(0xC0 | (code >> 6));
                    utf8 += static_cast<char>(0x80 | (code & 0x3F));
                } else {
                    utf8 += static_cast<char>(0xE0 | ((code >> 12) & 0x0F));
                    utf8 += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                    utf8 += static_cast<char>(0x80 | (code & 0x3F));
                }
                s.replace(pos, end - pos + 1, utf8);
                pos += utf8.size();
            }
            return s;
        }

        using Attrs = std::map<std::string, std::string>;

        // Attributes of one start tag: the text between its name and '>'.
        Attrs parseAttrs(const std::string& xml, std::size_t i, std::size_t end) {
            Attrs a;
            auto space = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
            while (i < end) {
                while (i < end && (space(xml[i]) || xml[i] == '/')) ++i;
                const std::size_t nameStart = i;
                while (i < end && xml[i] != '=' && !space(xml[i])) ++i;
                if (i >= end) break;
                std::string name = xml.substr(nameStart, i - nameStart);
                const std::size_t colon = name.find(':');
                if (colon != std::string::npos && name.compare(0, colon, "xmlns") != 0) name = name.substr(colon + 1);
                while (i < end && (space(xml[i]) || xml[i] == '=')) ++i;
                if (i >= end) break;
                const char quote = xml[i];
                if (quote != '"' && quote != '\'') {
                    ++i;
                    continue;
                }
                const std::size_t valueStart = ++i;
                const std::size_t valueEnd = xml.find(quote, valueStart);
                if (valueEnd == std::string::npos || valueEnd > end) break;
                a[name] = xmlUnescape(xml.substr(valueStart, valueEnd - valueStart));
                i = valueEnd + 1;
            }
            return a;
        }

        double attrDouble(const Attrs& a, const char* key, double def = 0.0) {
            const auto it = a.find(key);
            return it == a.end() ? def : toDouble(it->second, def);
        }
        std::string attrString(const Attrs& a, const char* key) {
            const auto it = a.find(key);
            return it == a.end() ? std::string() : it->second;
        }
        bool has(const Attrs& a, const char* key) { return a.find(key) != a.end(); }

        // One pass over the XML, keeping just enough nesting to put each
        // Pixels / Channel / TiffData / UUID with its Image.
        void parseOme(const std::string& xml, TiffMetadata& md) {
            md.ome = true;
            bool inPixels = false;
            std::size_t pos = 0;
            auto space = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
            while ((pos = xml.find('<', pos)) != std::string::npos) {
                if (xml.compare(pos, 4, "<!--") == 0) {
                    const std::size_t e = xml.find("-->", pos + 4);
                    if (e == std::string::npos) break;
                    pos = e + 3;
                    continue;
                }
                if (xml.compare(pos, 9, "<![CDATA[") == 0) {
                    const std::size_t e = xml.find("]]>", pos + 9);
                    if (e == std::string::npos) break;
                    pos = e + 3;
                    continue;
                }
                std::size_t i = pos + 1;
                if (i < xml.size() && (xml[i] == '?' || xml[i] == '!')) {
                    const std::size_t e = xml.find('>', i);
                    if (e == std::string::npos) break;
                    pos = e + 1;
                    continue;
                }
                const bool closing = i < xml.size() && xml[i] == '/';
                if (closing) ++i;
                std::size_t nameEnd = i;
                while (nameEnd < xml.size() && !space(xml[nameEnd]) && xml[nameEnd] != '>' && xml[nameEnd] != '/')
                    ++nameEnd;
                std::string name = xml.substr(i, nameEnd - i);
                const std::size_t colon = name.find(':');
                if (colon != std::string::npos) name = name.substr(colon + 1);
                // the closing '>' outside quoted attribute values
                std::size_t close = nameEnd;
                char quote = 0;
                for (; close < xml.size(); ++close) {
                    const char ch = xml[close];
                    if (quote) {
                        if (ch == quote) quote = 0;
                    } else if (ch == '"' || ch == '\'') {
                        quote = ch;
                    } else if (ch == '>') {
                        break;
                    }
                }
                if (close >= xml.size()) break;
                const bool selfClosing = !closing && close > nameEnd && xml[close - 1] == '/';
                pos = close + 1;

                if (closing) {
                    if (name == "Pixels") inPixels = false;
                    continue;
                }
                if (name == "OME") {
                    md.omeUuid = attrString(parseAttrs(xml, nameEnd, close), "UUID");
                } else if (name == "Image") {
                    const Attrs a = parseAttrs(xml, nameEnd, close);
                    OmeImage img;
                    img.id = attrString(a, "ID");
                    img.name = attrString(a, "Name");
                    md.omeImages.push_back(std::move(img));
                } else if (name == "Pixels") {
                    if (md.omeImages.empty()) md.omeImages.emplace_back();   // Pixels outside an Image: lenient
                    const Attrs a = parseAttrs(xml, nameEnd, close);
                    OmeImage& img = md.omeImages.back();
                    img.dimensionOrder = attrString(a, "DimensionOrder");
                    if (img.dimensionOrder.empty()) img.dimensionOrder = "XYZCT";   // OME schema default
                    img.type = attrString(a, "Type");
                    img.sizeX = toCount(attrDouble(a, "SizeX"));
                    img.sizeY = toCount(attrDouble(a, "SizeY"));
                    img.sizeZ = toCount(attrDouble(a, "SizeZ"));
                    img.sizeC = toCount(attrDouble(a, "SizeC"));
                    img.sizeT = toCount(attrDouble(a, "SizeT"));
                    img.physicalSizeUm = {attrDouble(a, "PhysicalSizeX") * tiffUnitToUm(attrString(a, "PhysicalSizeXUnit")),
                                          attrDouble(a, "PhysicalSizeY") * tiffUnitToUm(attrString(a, "PhysicalSizeYUnit")),
                                          attrDouble(a, "PhysicalSizeZ") * tiffUnitToUm(attrString(a, "PhysicalSizeZUnit"))};
                    img.timeIncrementS = attrDouble(a, "TimeIncrement") * timeUnitToS(attrString(a, "TimeIncrementUnit"));
                    img.interleaved = lower(attrString(a, "Interleaved")) == "true";
                    inPixels = !selfClosing;
                } else if (name == "Channel" && inPixels) {
                    const Attrs a = parseAttrs(xml, nameEnd, close);
                    OmeChannel ch;
                    ch.id = attrString(a, "ID");
                    ch.name = attrString(a, "Name");
                    const double spp = attrDouble(a, "SamplesPerPixel", 1.0);
                    ch.samplesPerPixel = spp >= 1.0 ? static_cast<std::uint32_t>(toCount(spp)) : 1u;
                    const auto wavelength = [&](const char* key, const char* unitKey) {
                        const double v = attrDouble(a, key);
                        const std::string unit = attrString(a, unitKey);
                        return v > 0 ? v * tiffUnitToUm(unit.empty() ? "nm" : unit) * 1e3 : 0.0;   // um -> nm
                    };
                    ch.emissionNm = wavelength("EmissionWavelength", "EmissionWavelengthUnit");
                    ch.excitationNm = wavelength("ExcitationWavelength", "ExcitationWavelengthUnit");
                    ch.fluor = attrString(a, "Fluor");
                    const std::string color = attrString(a, "Color");
                    if (!color.empty()) {
                        try {
                            const long long v = std::stoll(color);
                            ch.colorRgba = static_cast<std::uint32_t>(static_cast<std::int32_t>(v));
                            ch.hasColor = true;
                        } catch (...) {
                        }
                    }
                    md.omeImages.back().channels.push_back(std::move(ch));
                } else if (name == "TiffData" && inPixels) {
                    const Attrs a = parseAttrs(xml, nameEnd, close);
                    OmeTiffData td;
                    td.hasIfd = has(a, "IFD");
                    td.ifd = static_cast<std::uint32_t>(toCount(attrDouble(a, "IFD")));
                    td.firstC = static_cast<std::uint32_t>(toCount(attrDouble(a, "FirstC")));
                    td.firstZ = static_cast<std::uint32_t>(toCount(attrDouble(a, "FirstZ")));
                    td.firstT = static_cast<std::uint32_t>(toCount(attrDouble(a, "FirstT")));
                    td.hasPlaneCount = has(a, "PlaneCount");
                    td.planeCount = static_cast<std::uint32_t>(toCount(attrDouble(a, "PlaneCount")));
                    md.omeImages.back().tiffData.push_back(std::move(td));
                } else if (name == "UUID" && inPixels && !selfClosing) {
                    auto& blocks = md.omeImages.back().tiffData;
                    if (!blocks.empty()) {
                        const Attrs a = parseAttrs(xml, nameEnd, close);
                        blocks.back().fileName = attrString(a, "FileName");
                        const std::size_t textEnd = xml.find('<', pos);
                        if (textEnd != std::string::npos) blocks.back().uuid = trim(xmlUnescape(xml.substr(pos, textEnd - pos)));
                    }
                }
            }

            if (md.omeImages.empty()) return;
            const OmeImage& first = md.omeImages.front();
            md.sizeC = first.sizeC;
            md.sizeZ = first.sizeZ;
            md.sizeT = first.sizeT;
            md.dimensionOrder = first.dimensionOrder;
            md.voxelUm = first.physicalSizeUm;
            md.frameIntervalS = first.timeIncrementS;
            for (const OmeChannel& c : first.channels) {
                TiffChannel ch;
                ch.name = c.name;
                ch.emissionNm = c.emissionNm;
                if (c.hasColor) {
                    const std::uint32_t u = c.colorRgba;
                    ch.color = {static_cast<float>((u >> 24) & 0xFF) / 255.f, static_cast<float>((u >> 16) & 0xFF) / 255.f,
                                static_cast<float>((u >> 8) & 0xFF) / 255.f};
                    ch.hasColor = true;
                }
                md.channels.push_back(std::move(ch));
            }
        }

        void parseImageJ(const std::string& text, TiffMetadata& md) {
            md.imagej = true;
            ImageJMetadata& ij = md.imageJ;
            std::istringstream in(text);
            std::string line;
            while (std::getline(in, line)) {
                const std::size_t eq = line.find('=');
                if (eq == std::string::npos) continue;
                ij.entries[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
            }
            auto num = [&](const char* key, double def) {
                const auto it = ij.entries.find(key);
                return it == ij.entries.end() ? def : toDouble(it->second, def);
            };
            auto text_ = [&](const char* key) {
                const auto it = ij.entries.find(key);
                return it == ij.entries.end() ? std::string() : it->second;
            };
            ij.version = text_("ImageJ");
            ij.images = static_cast<std::uint32_t>(toCount(num("images", 0)));
            ij.channels = static_cast<std::uint32_t>(toCount(num("channels", 0)));
            ij.slices = static_cast<std::uint32_t>(toCount(num("slices", 0)));
            ij.frames = static_cast<std::uint32_t>(toCount(num("frames", 0)));
            ij.hyperstack = lower(text_("hyperstack")) == "true";
            ij.mode = text_("mode");
            ij.unit = text_("unit");
            ij.unitUm = tiffUnitToUm(ij.unit);
            ij.spacing = num("spacing", 0.0);
            ij.frameInterval = num("finterval", 0.0);
            ij.hasRange = ij.entries.count("min") && ij.entries.count("max");
            ij.min = num("min", 0.0);
            ij.max = num("max", 0.0);

            md.sizeC = ij.channels;
            md.sizeZ = ij.slices;
            md.sizeT = ij.frames;
            md.dimensionOrder = "XYCZT";   // hyperstacks: channel fastest, then slice, then frame
            if (ij.spacing > 0 && ij.unitUm > 0) md.voxelUm[2] = ij.spacing * ij.unitUm;
            md.frameIntervalS = ij.frameInterval;
        }

    } // namespace

    double tiffUnitToUm(const std::string& unitIn) {
        const std::string u = lower(trim(unitIn));
        if (u.empty() || u == "\xc2\xb5m" || u == "um" || u == "micron" || u == "microns" || u == "micrometer" ||
            u == "micrometre" || u == "\xce\xbcm" || u == "\\u00b5m")
            return 1.0;
        if (u == "nm" || u == "nanometer" || u == "nanometre") return 1e-3;
        if (u == "mm" || u == "millimeter" || u == "millimetre") return 1e3;
        if (u == "cm" || u == "centimeter" || u == "centimetre") return 1e4;
        if (u == "m" || u == "meter" || u == "metre") return 1e6;
        if (u == "inch" || u == "in") return 2.54e4;
        if (u == "pixel" || u == "pixels") return 0.0;
        return 1.0;
    }

    TiffMetadata parseTiffMetadata(const std::string& description) {
        TiffMetadata md;
        if (description.find("<OME") != std::string::npos || description.find("<ome") != std::string::npos)
            parseOme(description, md);
        else if (description.rfind("ImageJ=", 0) == 0 || description.find("\nImageJ=") != std::string::npos)
            parseImageJ(description, md);
        return md;
    }

    std::vector<std::vector<std::uint32_t>> omeImagePages(const TiffMetadata& md, std::size_t pageCount,
                                                          std::uint32_t samplesPerPixel) {
        std::vector<std::vector<std::uint32_t>> out;
        constexpr std::uint32_t kMissing = std::numeric_limits<std::uint32_t>::max();
        std::uint64_t nextPage = 0;   // where an image without TiffData starts
        const std::uint32_t spp = std::max<std::uint32_t>(samplesPerPixel, 1);
        for (const OmeImage& img : md.omeImages) {
            const std::uint64_t z = std::max<std::uint64_t>(img.sizeZ, 1), t = std::max<std::uint64_t>(img.sizeT, 1);
            std::uint64_t c = std::max<std::uint64_t>(img.sizeC, 1);
            if (spp > 1 && c % spp == 0) c /= spp;   // an RGB channel is one plane of spp samples
            const std::uint64_t planes = std::min<std::uint64_t>(c * z * t, pageCount);
            std::vector<std::uint32_t> pages(static_cast<std::size_t>(planes), kMissing);
            // plane index of (c, z, t) in DimensionOrder (after XY, fastest first)
            std::string order;
            for (char ch : img.dimensionOrder) {
                const char l = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
                if ((l == 'C' || l == 'Z' || l == 'T') && order.find(l) == std::string::npos) order += l;
            }
            for (char l : std::string("ZCT"))   // OME default XYZCT: Z fastest after XY
                if (order.find(l) == std::string::npos) order += l;
            auto planeOf = [&](std::uint64_t ci, std::uint64_t zi, std::uint64_t ti) {
                std::uint64_t idx = 0, stride = 1;
                for (char l : order) {
                    const std::uint64_t v = l == 'C' ? ci : l == 'Z' ? zi
                                                                     : ti;
                    const std::uint64_t n = l == 'C' ? c : l == 'Z' ? z
                                                                    : t;
                    idx += v * stride;
                    stride *= n;
                }
                return idx;
            };
            std::uint64_t last = nextPage;
            if (img.tiffData.empty()) {
                for (std::uint64_t p = 0; p < planes && nextPage + p < pageCount; ++p) {
                    pages[static_cast<std::size_t>(p)] = static_cast<std::uint32_t>(nextPage + p);
                    last = nextPage + p + 1;
                }
            } else {
                for (const OmeTiffData& td : img.tiffData) {
                    if (!td.uuid.empty() && !md.omeUuid.empty() && td.uuid != md.omeUuid) continue;   // another file
                    if (!td.uuid.empty() && md.omeUuid.empty()) continue;
                    const std::uint64_t count = td.hasPlaneCount ? td.planeCount : (td.hasIfd ? 1 : planes);
                    const std::uint64_t start = planeOf(td.firstC, td.firstZ, td.firstT);
                    for (std::uint64_t k = 0; k < count; ++k) {
                        const std::uint64_t plane = start + k, page = std::uint64_t{td.ifd} + k;
                        if (plane >= planes || page >= pageCount) break;
                        pages[static_cast<std::size_t>(plane)] = static_cast<std::uint32_t>(page);
                        last = std::max(last, page + 1);
                    }
                }
            }
            const auto missing = std::find(pages.begin(), pages.end(), kMissing);
            pages.erase(missing, pages.end());
            nextPage = last;
            out.push_back(std::move(pages));
        }
        return out;
    }

} // namespace sirius
