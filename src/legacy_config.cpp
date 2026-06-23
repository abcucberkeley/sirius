#include "sirius/legacy_config.hpp"

#include <cctype>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius {

    namespace {

        std::string trim(const std::string& s) {
            const auto* ws = " \t\r\n";
            const auto b = s.find_first_not_of(ws);
            if (b == std::string::npos) return {};
            const auto e = s.find_last_not_of(ws);
            return s.substr(b, e - b + 1);
        }

        // Typed value parsers. `key` is only used to make errors actionable.
        int parseInt(const std::string& key, const std::string& v) {
            try {
                size_t pos = 0;
                int out = std::stoi(v, &pos);
                if (pos != v.size()) throw std::invalid_argument(v);
                return out;
            } catch (const std::exception&) {
                throw std::runtime_error("config key '" + key + "' expects an integer, got: " + v);
            }
        }

        float parseFloat(const std::string& key, const std::string& v) {
            try {
                size_t pos = 0;
                float out = std::stof(v, &pos);
                if (pos != v.size()) throw std::invalid_argument(v);
                return out;
            } catch (const std::exception&) {
                throw std::runtime_error("config key '" + key + "' expects a number, got: " + v);
            }
        }

        bool parseBool(const std::string& key, const std::string& v) {
            if (v == "1" || v == "true"  || v == "True")  return true;
            if (v == "0" || v == "false" || v == "False") return false;
            throw std::runtime_error("config key '" + key + "' expects 0/1, got: " + v);
        }

        std::vector<float> parseFloatList(const std::string& key, const std::string& v) {
            std::vector<float> out;
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ',')) {
                item = trim(item);
                if (!item.empty()) out.push_back(parseFloat(key, item));
            }
            return out;
        }

        using Setter = std::function<void(LegacyReconConfig&, const std::string& key, const std::string& val)>;

        // Single source of truth for recognized legacy keys. Add an alias here.
        const std::unordered_map<std::string, Setter>& aliasTable() {
            static const std::unordered_map<std::string, Setter> table = {
                // geometry / optics
                {"ndirs",      [](auto& c, auto& k, auto& v){ c.ndirs = parseInt(k, v); }},
                {"nphases",    [](auto& c, auto& k, auto& v){ c.nphases = parseInt(k, v); }},
                {"nordersout", [](auto& c, auto& k, auto& v){ c.norders_output = parseInt(k, v); }},
                {"norders",    [](auto& c, auto& k, auto& v){ c.norders = parseInt(k, v); }},
                {"nbeams",     [](auto& c, auto& k, auto& v){ c.nbeams = parseInt(k, v); }},
                {"ls",         [](auto& c, auto& k, auto& v){ c.linespacing = parseFloat(k, v); }},
                {"angle0",     [](auto& c, auto& k, auto& v){ c.k0startangle = parseFloat(k, v); }},
                {"k0angles",   [](auto& c, auto& k, auto& v){ c.k0angles = parseFloatList(k, v); }},
                {"na",         [](auto& c, auto& k, auto& v){ c.na = parseFloat(k, v); }},
                {"nimm",       [](auto& c, auto& k, auto& v){ c.nimm = parseFloat(k, v); }},
                {"wavelength", [](auto& c, auto& k, auto& v){ c.wavelengthNm = parseFloat(k, v); }},

                // pixel sizes
                {"xyres",      [](auto& c, auto& k, auto& v){ c.dxy = parseFloat(k, v); }},
                {"zres",       [](auto& c, auto& k, auto& v){ c.dz = parseFloat(k, v); }},
                {"zresPSF",    [](auto& c, auto& k, auto& v){ c.dzPSF = parseFloat(k, v); }},

                // I5S / Bessel / deskew
                {"2lenses",       [](auto& c, auto& k, auto& v){ c.bTwolens = parseBool(k, v); }},
                {"bessel",        [](auto& c, auto& k, auto& v){ c.bBessel = parseBool(k, v); }},
                {"besselNA",      [](auto& c, auto& k, auto& v){ c.BesselNA = parseFloat(k, v); }},
                {"besselLambdaEx",[](auto& c, auto& k, auto& v){ c.BesselLambdaEx = parseFloat(k, v); }},
                {"deskew",        [](auto& c, auto& k, auto& v){ c.deskewAngle = parseFloat(k, v); }},
                {"deskewshift",   [](auto& c, auto& k, auto& v){ c.extraShift = parseInt(k, v); }},
                {"noRecon",       [](auto& c, auto& k, auto& v){ c.bNoRecon = parseBool(k, v); }},
                {"cropXY",        [](auto& c, auto& k, auto& v){ c.cropXYto = parseInt(k, v); }},
                {"writeTitle",    [](auto& c, auto& k, auto& v){ c.bWriteTitle = parseBool(k, v); }},

                // algorithm / filtering
                {"otfcutoff",   [](auto& c, auto& k, auto& v){ c.otfcutoff = parseFloat(k, v); }},
                {"zoomfact",    [](auto& c, auto& k, auto& v){ c.zoomfact = parseFloat(k, v); }},
                {"zzoom",       [](auto& c, auto& k, auto& v){ c.z_zoom = parseInt(k, v); }},
                {"nzPadTo",     [](auto& c, auto& k, auto& v){ c.nzPadTo = parseInt(k, v); }},
                {"explodefact", [](auto& c, auto& k, auto& v){ c.explodefact = parseFloat(k, v); }},
                {"nofilteroverlaps", [](auto& c, auto& k, auto& v){ c.bFilteroverlaps = !parseBool(k, v); }},
                {"recalcarrays",[](auto& c, auto& k, auto& v){ c.recalcarrays = parseInt(k, v); }},
                {"napodize",    [](auto& c, auto& k, auto& v){ c.napodize = parseInt(k, v); }},
                {"searchforvector", [](auto& c, auto& k, auto& v){ c.bSearchforvector = parseInt(k, v); }},
                {"usetime0k0",  [](auto& c, auto& k, auto& v){ c.bUseTime0k0 = parseInt(k, v); }},
                {"apodizeoutput",[](auto& c, auto& k, auto& v){ c.apodizeoutput = parseInt(k, v); }},
                {"gammaApo",    [](auto& c, auto& k, auto& v){ c.apoGamma = parseFloat(k, v); }},
                {"nosuppress",  [](auto& c, auto& k, auto& v){ c.bSuppress_singularities = parseBool(k, v) ? 0 : 1; }},
                {"suppressR",   [](auto& c, auto& k, auto& v){ c.suppression_radius = parseInt(k, v); }},
                {"dampenOrder0",[](auto& c, auto& k, auto& v){ c.bDampenOrder0 = parseBool(k, v); }},
                {"fitallphases",[](auto& c, auto& k, auto& v){ c.bFitallphases = parseInt(k, v); }},
                {"norescale",   [](auto& c, auto& k, auto& v){ c.do_rescale = parseBool(k, v) ? 0 : 1; }},
                {"equalizez",   [](auto& c, auto& k, auto& v){ c.equalizez = parseBool(k, v); }},
                {"equalizet",   [](auto& c, auto& k, auto& v){ c.equalizet = parseBool(k, v); }},
                {"nokz0",       [](auto& c, auto& k, auto& v){ c.bNoKz0 = parseBool(k, v); }},
                {"wiener",      [](auto& c, auto& k, auto& v){ c.wiener = parseFloat(k, v); }},
                {"wienerInr",   [](auto& c, auto& k, auto& v){ c.wienerInr = parseFloat(k, v); }},
                {"fastSI",      [](auto& c, auto& k, auto& v){ c.bFastSIM = parseBool(k, v); }},
                {"forcemodamp", [](auto& c, auto& k, auto& v){ c.forceamp = parseFloatList(k, v); }},
                {"phaseSteps",  [](auto& c, auto& k, auto& v){ c.phaseSteps = parseFloatList(k, v); }},

                // OTF geometry
                {"otfRA",       [](auto& c, auto& k, auto& v){ c.bRadAvgOTF = parseBool(k, v); }},
                {"otfPerAngle", [](auto& c, auto& k, auto& v){ c.bOneOTFperAngle = parseBool(k, v); }},
                {"nxotf",       [](auto& c, auto& k, auto& v){ c.nxotf = parseInt(k, v); }},
                {"nyotf",       [](auto& c, auto& k, auto& v){ c.nyotf = parseInt(k, v); }},
                {"nzotf",       [](auto& c, auto& k, auto& v){ c.nzotf = parseInt(k, v); }},
                {"dkrotf",      [](auto& c, auto& k, auto& v){ c.dkrotf = parseFloat(k, v); }},
                {"dkzotf",      [](auto& c, auto& k, auto& v){ c.dkzotf = parseFloat(k, v); }},

                // drift
                {"fixdrift",          [](auto& c, auto& k, auto& v){ c.bFixdrift = parseInt(k, v); }},
                {"drift_filter_fact", [](auto& c, auto& k, auto& v){ c.drift_filter_fact = parseFloat(k, v); }},

                // camera
                {"background",       [](auto& c, auto& k, auto& v){ c.constbkgd = parseFloat(k, v); }},
                {"bgInExtHdr",       [](auto& c, auto& k, auto& v){ c.bBgInExtHdr = parseInt(k, v); }},
                {"usecorr",          [](auto& c, auto& k, auto& v){ c.corrfiles = v; c.bUsecorr = 1; }},
                {"readoutNoiseVar",  [](auto& c, auto& k, auto& v){ c.readoutNoiseVar = parseFloat(k, v); }},
                {"electrons_per_bit",[](auto& c, auto& k, auto& v){ c.electrons_per_bit = parseFloat(k, v); }},

                // debugging / intermediate output
                {"makemodel",     [](auto& c, auto& k, auto& v){ c.bMakemodel = parseInt(k, v); }},
                {"saveprefiltered",[](auto& c, auto& k, auto& v){ c.fileSeparated = v; c.bSaveSeparated = 1; }},
                {"savealignedraw",[](auto& c, auto& k, auto& v){ c.fileRawAligned = v; c.bSaveAlignedRaw = 1; }},
                {"saveoverlaps",  [](auto& c, auto& k, auto& v){ c.fileOverlaps = v; c.bSaveOverlaps = 1; }},

                // I/O
                {"input",  [](auto& c, auto& k, auto& v){ c.ifiles = v; }},
                {"output", [](auto& c, auto& k, auto& v){ c.ofiles = v; }},
                {"otf",    [](auto& c, auto& k, auto& v){ c.otffiles = v; }},
            };
            return table;
        }

    } // namespace

    LegacyReconConfig loadLegacyConfig(const std::string& path) {
        std::ifstream file(path);
        if (!file)
            throw std::runtime_error("Failed to open legacy config: " + path);

        LegacyReconConfig c;
        const auto& table = aliasTable();

        std::string line;
        int lineNo = 0;
        while (std::getline(file, line)) {
            ++lineNo;
            const std::string s = trim(line);
            if (s.empty() || s[0] == '#' || s[0] == ';')
                continue;

            const auto eq = s.find('=');
            if (eq == std::string::npos)
                throw std::runtime_error("Malformed line " + std::to_string(lineNo) +
                                         " in " + path + " (expected key=value): " + s);

            const std::string key = trim(s.substr(0, eq));
            const std::string val = trim(s.substr(eq + 1));

            const auto it = table.find(key);
            if (it == table.end())
                throw std::runtime_error("Unknown legacy config key '" + key +
                                         "' on line " + std::to_string(lineNo) + " of " + path);
            it->second(c, key, val);
        }
        return c;
    }

    SIMParameters fromLegacy(const LegacyReconConfig& c) {
        SIMParameters p;

        p.ndirs          = c.ndirs;
        p.nphases        = c.nphases;
        p.linespacing_um = c.linespacing;
        p.k0_start_angle = c.k0startangle;
        p.na             = c.na;
        p.nimm           = c.nimm;
        p.wavelength_nm  = c.wavelengthNm;
        if (!c.k0angles.empty())
            p.k0_angles = std::vector<double>(c.k0angles.begin(), c.k0angles.end());

        p.dx     = c.dxy;
        p.dy     = c.dxy;
        p.dz     = c.dz;
        p.dz_psf = c.dzPSF;

        p.zoomfact               = c.zoomfact;
        p.z_zoom                 = c.z_zoom;
        p.wiener                 = c.wiener;
        p.otfcutoff              = c.otfcutoff;
        p.background             = c.constbkgd;
        p.suppression_radius     = c.suppression_radius;
        p.suppress_singularities = (c.bSuppress_singularities != 0);
        p.dampen_order0          = c.bDampenOrder0;
        p.explodefact            = c.explodefact;
        p.fast_si                = c.bFastSIM;
        p.do_rescale             = (c.do_rescale != 0);
        p.equalizez              = c.equalizez;
        p.no_kz0                 = c.bNoKz0;
        p.filter_overlaps        = c.bFilteroverlaps;

        // Legacy decodes the input apodization from napodize at runtime in
        // apodizationDriver(): >0 => edge ("triangle") blend of that width,
        // exactly -1 => cosine window, anything else (0 or other negatives)
        // => no apodization.
        if (c.napodize > 0) {
            p.apodize_input = ApodizationType::Triangle;
            p.napodize      = c.napodize;
        } else if (c.napodize == -1) {
            p.apodize_input = ApodizationType::Cosine;
            p.napodize      = 0;            // width is meaningless for the cosine window
        } else {
            p.apodize_input = ApodizationType::None;
            p.napodize      = 0;
        }

        switch (c.apodizeoutput) {
            case 0: p.apodize_output = ApodizationType::None;     break;
            case 1: p.apodize_output = ApodizationType::Cosine;   break;
            case 2: p.apodize_output = ApodizationType::Triangle; break;
            default:
                throw std::runtime_error("apodizeoutput must be 0, 1, or 2, got: " +
                                         std::to_string(c.apodizeoutput));
        }

        p.validate();
        return p;
    }

} // namespace sirius
