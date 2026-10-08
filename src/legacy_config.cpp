#include "sirius/legacy_config.hpp"
#include "sirius/errors.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <optional>
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
                throw IoError("config key '" + key + "' expects an integer, got: " + v);
            }
        }

        float parseFloat(const std::string& key, const std::string& v) {
            try {
                size_t pos = 0;
                float out = std::stof(v, &pos);
                if (pos != v.size()) throw std::invalid_argument(v);
                return out;
            } catch (const std::exception&) {
                throw IoError("config key '" + key + "' expects a number, got: " + v);
            }
        }

        double parseDouble(const std::string& key, const std::string& v) {
            try {
                size_t pos = 0;
                double out = std::stod(v, &pos);
                if (pos != v.size()) throw std::invalid_argument(v);
                return out;
            } catch (const std::exception&) {
                throw IoError("config key '" + key + "' expects a number, got: " + v);
            }
        }

        bool parseBool(const std::string& key, const std::string& v) {
            if (v == "1" || v == "true" || v == "True") return true;
            if (v == "0" || v == "false" || v == "False") return false;
            throw IoError("config key '" + key + "' expects 0/1, got: " + v);
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
                {"ndirs", [](auto& c, auto& k, auto& v) { c.ndirs = parseInt(k, v); }},
                {"nphases", [](auto& c, auto& k, auto& v) { c.nphases = parseInt(k, v); }},
                {"nordersout", [](auto& c, auto& k, auto& v) { c.norders_output = parseInt(k, v); }},
                {"norders", [](auto& c, auto& k, auto& v) { c.norders = parseInt(k, v); }},
                {"nbeams", [](auto& c, auto& k, auto& v) { c.nbeams = parseInt(k, v); }},
                {"ls", [](auto& c, auto& k, auto& v) { c.linespacing = parseFloat(k, v); }},
                {"angle0", [](auto& c, auto& k, auto& v) { c.k0startangle = parseFloat(k, v); }},
                {"k0angles", [](auto& c, auto& k, auto& v) { c.k0angles = parseFloatList(k, v); }},
                {"na", [](auto& c, auto& k, auto& v) { c.na = parseFloat(k, v); }},
                {"nimm", [](auto& c, auto& k, auto& v) { c.nimm = parseFloat(k, v); }},
                {"wavelength", [](auto& c, auto& k, auto& v) { c.wavelengthNm = parseFloat(k, v); }},

                // pixel sizes
                {"xyres", [](auto& c, auto& k, auto& v) { c.dxy = parseDouble(k, v); }},
                {"zres", [](auto& c, auto& k, auto& v) { c.dz = parseDouble(k, v); }},
                {"zresPSF", [](auto& c, auto& k, auto& v) { c.dzPSF = parseDouble(k, v); }},

                // I5S / Bessel / deskew
                {"2lenses", [](auto& c, auto& k, auto& v) { c.bTwolens = parseBool(k, v); }},
                {"bessel", [](auto& c, auto& k, auto& v) { c.bBessel = parseBool(k, v); }},
                {"besselNA", [](auto& c, auto& k, auto& v) { c.BesselNA = parseFloat(k, v); }},
                {"besselLambdaEx", [](auto& c, auto& k, auto& v) { c.BesselLambdaEx = parseFloat(k, v); }},
                // cudasirecon's own name for the same quantity (microns); both spellings are read
                {"besselExWave", [](auto& c, auto& k, auto& v) { c.BesselLambdaEx = parseFloat(k, v); }},
                {"deskew", [](auto& c, auto& k, auto& v) { c.deskewAngle = parseFloat(k, v); }},
                {"deskewshift", [](auto& c, auto& k, auto& v) { c.extraShift = parseInt(k, v); }},
                {"noRecon", [](auto& c, auto& k, auto& v) { c.bNoRecon = parseBool(k, v); }},
                {"cropXY", [](auto& c, auto& k, auto& v) { c.cropXYto = parseInt(k, v); }},
                {"writeTitle", [](auto& c, auto& k, auto& v) { c.bWriteTitle = parseBool(k, v); }},

                // algorithm / filtering
                {"otfcutoff", [](auto& c, auto& k, auto& v) { c.otfcutoff = parseFloat(k, v); }},
                {"zoomfact", [](auto& c, auto& k, auto& v) { c.zoomfact = parseFloat(k, v); }},
                {"zzoom", [](auto& c, auto& k, auto& v) { c.z_zoom = parseInt(k, v); }},
                {"nzPadTo", [](auto& c, auto& k, auto& v) { c.nzPadTo = parseInt(k, v); }},
                {"explodefact", [](auto& c, auto& k, auto& v) { c.explodefact = parseFloat(k, v); }},
                // 1.1.1 spells it `nofilteroverlaps`; 1.2.0 and the user's
                // fork spell the same switch `nofilterovlps`, and SIRIUS knew
                // only the first, so the fork's own vocabulary was refused.
                {"nofilteroverlaps", [](auto& c, auto& k, auto& v) { c.bFilteroverlaps = !parseBool(k, v); }},
                {"nofilterovlps", [](auto& c, auto& k, auto& v) { c.bFilteroverlaps = !parseBool(k, v); }},
                {"recalcarrays", [](auto& c, auto& k, auto& v) { c.recalcarrays = parseInt(k, v); }},
                {"napodize", [](auto& c, auto& k, auto& v) { c.napodize = parseInt(k, v); }},
                {"searchforvector", [](auto& c, auto& k, auto& v) { c.bSearchforvector = parseInt(k, v); }},
                {"usetime0k0", [](auto& c, auto& k, auto& v) { c.bUseTime0k0 = parseInt(k, v); }},
                {"k0searchAll", [](auto& c, auto& k, auto& v) { c.k0searchAll = parseInt(k, v); }},
                // a cudasirecon CLI flag (print the version); harmless in a parameter file
                {"version", [](auto&, auto&, auto&) {}},
                {"apodizeoutput", [](auto& c, auto& k, auto& v) { c.apodizeoutput = parseInt(k, v); }},
                {"gammaApo", [](auto& c, auto& k, auto& v) { c.apoGamma = parseFloat(k, v); }},
                {"nosuppress", [](auto& c, auto& k, auto& v) { c.bSuppress_singularities = parseBool(k, v) ? 0 : 1; }},
                {"suppressR", [](auto& c, auto& k, auto& v) { c.suppression_radius = parseInt(k, v); }},
                {"dampenOrder0", [](auto& c, auto& k, auto& v) { c.bDampenOrder0 = parseBool(k, v); }},
                {"fitallphases", [](auto& c, auto& k, auto& v) { c.bFitallphases = parseInt(k, v); }},
                {"norescale", [](auto& c, auto& k, auto& v) { c.do_rescale = parseBool(k, v) ? 0 : 1; }},
                {"equalizez", [](auto& c, auto& k, auto& v) { c.equalizez = parseBool(k, v); }},
                {"equalizet", [](auto& c, auto& k, auto& v) { c.equalizet = parseBool(k, v); }},
                {"nokz0", [](auto& c, auto& k, auto& v) { c.bNoKz0 = parseBool(k, v); }},
                {"wiener", [](auto& c, auto& k, auto& v) { c.wiener = parseFloat(k, v); }},
                {"wienerInr", [](auto& c, auto& k, auto& v) { c.wienerInr = parseFloat(k, v); }},
                {"fastSI", [](auto& c, auto& k, auto& v) { c.bFastSIM = parseBool(k, v); }},
                {"forcemodamp", [](auto& c, auto& k, auto& v) { c.forceamp = parseFloatList(k, v); }},
                {"phaseSteps", [](auto& c, auto& k, auto& v) { c.phaseSteps = parseFloatList(k, v); }},

                // OTF geometry
                {"otfRA", [](auto& c, auto& k, auto& v) { c.bRadAvgOTF = parseBool(k, v); }},
                {"otfPerAngle", [](auto& c, auto& k, auto& v) { c.bOneOTFperAngle = parseBool(k, v); }},
                {"nxotf", [](auto& c, auto& k, auto& v) { c.nxotf = parseInt(k, v); }},
                {"nyotf", [](auto& c, auto& k, auto& v) { c.nyotf = parseInt(k, v); }},
                {"nzotf", [](auto& c, auto& k, auto& v) { c.nzotf = parseInt(k, v); }},
                {"dkrotf", [](auto& c, auto& k, auto& v) { c.dkrotf = parseFloat(k, v); }},
                {"dkzotf", [](auto& c, auto& k, auto& v) { c.dkzotf = parseFloat(k, v); }},

                // drift
                {"fixdrift", [](auto& c, auto& k, auto& v) { c.bFixdrift = parseInt(k, v); }},
                {"drift_filter_fact", [](auto& c, auto& k, auto& v) { c.drift_filter_fact = parseFloat(k, v); }},

                // camera
                {"background", [](auto& c, auto& k, auto& v) { c.constbkgd = parseFloat(k, v); }},
                {"bgInExtHdr", [](auto& c, auto& k, auto& v) { c.bBgInExtHdr = parseInt(k, v); }},
                {"usecorr", [](auto& c, auto&, auto& v) { c.corrfiles = v; c.bUsecorr = 1; }},
                {"readoutNoiseVar", [](auto& c, auto& k, auto& v) { c.readoutNoiseVar = parseFloat(k, v); }},
                {"electrons_per_bit", [](auto& c, auto& k, auto& v) { c.electrons_per_bit = parseFloat(k, v); }},

                // debugging / intermediate output
                {"makemodel", [](auto& c, auto& k, auto& v) { c.bMakemodel = parseInt(k, v); }},
                {"saveprefiltered", [](auto& c, auto&, auto& v) { c.fileSeparated = v; c.bSaveSeparated = 1; }},
                {"savealignedraw", [](auto& c, auto&, auto& v) { c.fileRawAligned = v; c.bSaveAlignedRaw = 1; }},
                {"saveoverlaps", [](auto& c, auto&, auto& v) { c.fileOverlaps = v; c.bSaveOverlaps = 1; }},

                // I/O
                {"input", [](auto& c, auto&, auto& v) { c.ifiles = v; }},
                {"output", [](auto& c, auto&, auto& v) { c.ofiles = v; }},
                {"otf", [](auto& c, auto&, auto& v) { c.otffiles = v; }},

                // cudasirecon 1.2.0's output and tiling keys. Every one of
                // these appears in a config the user runs daily and in no
                // earlier cudasirecon, and the parser threw on the first of
                // them, so the whole file was refused.
                {"uint16", [](auto& c, auto& k, auto& v) { c.bUint16Output = parseBool(k, v); }},
                {"uint16offset", [](auto& c, auto& k, auto& v) { c.uint16Offset = parseFloat(k, v); }},
                {"cropXmin", [](auto& c, auto& k, auto& v) { c.cropXmin = parseInt(k, v); }},
                {"cropXmax", [](auto& c, auto& k, auto& v) { c.cropXmax = parseInt(k, v); }},
                {"cropYmin", [](auto& c, auto& k, auto& v) { c.cropYmin = parseInt(k, v); }},
                {"cropYmax", [](auto& c, auto& k, auto& v) { c.cropYmax = parseInt(k, v); }},
                {"cropZmin", [](auto& c, auto& k, auto& v) { c.cropZmin = parseInt(k, v); }},
                {"cropZmax", [](auto& c, auto& k, auto& v) { c.cropZmax = parseInt(k, v); }},
                {"chunkX", [](auto& c, auto& k, auto& v) { c.chunkX = parseInt(k, v); }},
                {"chunkY", [](auto& c, auto& k, auto& v) { c.chunkY = parseInt(k, v); }},
                {"chunkZ", [](auto& c, auto& k, auto& v) { c.chunkZ = parseInt(k, v); }},
                {"chunkOverlap", [](auto& c, auto& k, auto& v) { c.chunkOverlap = parseInt(k, v); }},
            };
            return table;
        }

        // What fromLegacy() does with each recognized key, and why, when the
        // answer is not "applies it". Accepting a key is not applying it: the
        // parser stores every key on LegacyReconConfig, and fromLegacy maps
        // only part of that onto SIMParameters. Before this table existed, the
        // difference was invisible -- a config saying gammaApo=0.5 or
        // fitallphases=0 or searchforvector=0 was read without complaint and
        // then reconstructed with SIRIUS's own value instead.
        //
        // Keeping it next to aliasTable() is deliberate: legacyConfigKeys()
        // and a test compare the two key sets, so a new alias without a status
        // fails the suite rather than being reported as applied by default.
        struct KeyStatus {
            LegacyKeyStatus status;
            const char*     reason;   // "" for Applied
        };

        const std::unordered_map<std::string, KeyStatus>& statusTable() {
            constexpr auto A = LegacyKeyStatus::Applied;
            constexpr auto P = LegacyKeyStatus::Pending;
            constexpr auto D = LegacyKeyStatus::Dropped;
            static const std::unordered_map<std::string, KeyStatus> table = {
                // --- in effect ------------------------------------------------
                {"ndirs", {A, ""}},
                {"nphases", {A, ""}},
                {"nordersout", {A, ""}},
                {"norders", {A, ""}},
                {"ls", {A, ""}},
                {"angle0", {A, ""}},
                {"k0angles", {A, ""}},
                {"na", {A, ""}},
                {"nimm", {A, ""}},
                {"wavelength", {A, ""}},
                {"xyres", {A, ""}},
                {"zres", {A, ""}},
                {"zresPSF", {A, ""}},
                {"otfcutoff", {A, ""}},
                {"zoomfact", {A, ""}},
                {"zzoom", {A, ""}},
                {"explodefact", {A, ""}},
                {"nofilteroverlaps", {A, ""}},
                {"nofilterovlps", {A, ""}},
                {"napodize", {A, ""}},
                {"apodizeoutput", {A, ""}},
                {"nosuppress", {A, ""}},
                {"suppressR", {A, ""}},
                {"dampenOrder0", {A, ""}},
                {"norescale", {A, ""}},
                {"equalizez", {A, ""}},
                {"nokz0", {A, ""}},
                {"wiener", {A, ""}},
                {"fastSI", {A, ""}},
                {"forcemodamp", {A, ""}},
                {"phaseSteps", {A, ""}},
                {"background", {A, ""}},

                // --- carried onto SIMParameters, not yet acted on -------------
                {"searchforvector",
                 {P, "mapped to SIMParameters::search_pattern_vector, which the "
                     "reconstruction does not read yet: k0 is fitted even when the file "
                     "says it is known"}},

                // --- parsed so the file loads, then thrown away ---------------
                {"gammaApo",
                 {D, "output apodization gamma: SIMParameters has no gamma, so the "
                     "reconstruction always uses 1 (plain triangular apodization)"}},
                {"fitallphases",
                 {D, "SIRIUS always uses every fitted phase; cudasirecon's 0 infers "
                     "order 2 and above from order 1's phase"}},
                {"equalizet",
                 {D, "bleach correction across time: SIRIUS reconstructs one (c, t) "
                     "volume at a time, so there is no series to equalize here"}},
                {"wienerInr", {D, "the per-order Wiener increment is not modelled"}},
                {"nbeams", {D, "the beam count is not modelled; the order count comes from nphases/norders"}},
                {"recalcarrays", {D, "a cudasirecon caching strategy with no counterpart"}},
                {"usetime0k0", {D, "SIRIUS fits each (c, t) volume it is given, so there is no time 0 to reuse"}},
                {"k0searchAll", {D, "SIRIUS already fits k0 on every volume it reconstructs, which is what this asks for"}},
                {"nzPadTo", {D, "axial padding of the input is not modelled"}},
                {"2lenses", {D, "I5S (two-objective) data is not supported"}},
                {"bessel", {D, "Bessel-SIM is not supported"}},
                {"besselNA", {D, "Bessel-SIM is not supported"}},
                {"besselLambdaEx", {D, "Bessel-SIM is not supported"}},
                {"besselExWave", {D, "Bessel-SIM is not supported"}},
                {"deskew", {D, "deskewing is a preprocessing step, not a reconstruction parameter"}},
                {"deskewshift", {D, "deskewing is a preprocessing step, not a reconstruction parameter"}},
                {"noRecon", {D, "whether to reconstruct is the caller's decision, not a parameter of the reconstruction"}},
                {"writeTitle", {D, "writes the command line into an MRC header; SIRIUS does not write MRC"}},
                {"otfRA", {D, "SIRIUS reads the radially averaged table the OTF file itself declares"}},
                {"otfPerAngle", {D, "one OTF per angle is not supported; one table serves every direction"}},
                {"nxotf", {D, "the OTF file carries its own geometry (otf_io.cpp); a config's copy is ignored"}},
                {"nyotf", {D, "the OTF file carries its own geometry (otf_io.cpp); a config's copy is ignored"}},
                {"nzotf", {D, "the OTF file carries its own geometry (otf_io.cpp); a config's copy is ignored"}},
                {"dkrotf", {D, "the OTF sampling is derived from the pixel sizes, not read from the config"}},
                {"dkzotf", {D, "the OTF sampling is derived from the pixel sizes, not read from the config"}},
                {"fixdrift", {D, "drift correction between directions is not implemented"}},
                {"drift_filter_fact", {D, "drift correction between directions is not implemented"}},
                {"bgInExtHdr", {D, "a per-frame background in an MRC extended header; SIRIUS does not read MRC headers here"}},
                {"usecorr", {D, "flat-field correction is a separate operation; pass the field to it"}},
                {"readoutNoiseVar", {D, "camera noise modelling is not implemented"}},
                {"electrons_per_bit", {D, "camera gain is not modelled"}},
                {"makemodel", {D, "a cudasirecon debug output"}},
                {"saveprefiltered", {D, "a cudasirecon debug output; SIRIUS captures bands through setCaptureDiagnostics"}},
                {"savealignedraw", {D, "a cudasirecon debug output; SIRIUS captures bands through setCaptureDiagnostics"}},
                {"saveoverlaps", {D, "a cudasirecon debug output; SIRIUS captures bands through setCaptureDiagnostics"}},
                {"input", {D, "a file path: the caller opens the dataset"}},
                {"output", {D, "a file path: the caller writes the result"}},
                {"otf", {D, "a file path: the caller passes the OTF to the step"}},
                {"version", {D, "a cudasirecon CLI flag (print the version); it is not a parameter"}},

                // 1.2.0's I/O and tiling keys. All of them describe what to
                // read and how to schedule it, so none belongs in
                // SIMParameters; see the report notes for where each goes.
                {"uint16", {D, "writes the output TIFF as uint16 instead of float32: an exporter option, not a reconstruction parameter"}},
                {"uint16offset", {D, "an offset added before the uint16 clamp: an exporter option, not a reconstruction parameter"}},
                {"cropXY", {D, "crop the lateral dimensions to a size: the caller crops before the step runs"}},
                {"cropXmin", {D, "crop bounding box (0-indexed, inclusive): the caller crops the dataset before the step runs"}},
                {"cropXmax", {D, "crop bounding box (0-indexed, inclusive): the caller crops the dataset before the step runs"}},
                {"cropYmin", {D, "crop bounding box (0-indexed, inclusive): the caller crops the dataset before the step runs"}},
                {"cropYmax", {D, "crop bounding box (0-indexed, inclusive): the caller crops the dataset before the step runs"}},
                {"cropZmin", {D, "crop bounding box in logical z (0-indexed, inclusive): the caller crops the dataset before the step runs"}},
                {"cropZmax", {D, "crop bounding box in logical z (0-indexed, inclusive): the caller crops the dataset before the step runs"}},
                {"chunkX", {D, "a tiling schedule for a stack too large to reconstruct in one piece: the caller tiles and stitches"}},
                {"chunkY", {D, "a tiling schedule for a stack too large to reconstruct in one piece: the caller tiles and stitches"}},
                {"chunkZ", {D, "a tiling schedule for a stack too large to reconstruct in one piece: the caller tiles and stitches"}},
                {"chunkOverlap", {D, "the overlap of the tiling schedule: the caller tiles and stitches"}},
            };
            return table;
        }

    } // namespace

    std::vector<std::string> LegacyConversionReport::notInEffect() const {
        std::vector<std::string> out = pending;
        out.insert(out.end(), dropped.begin(), dropped.end());
        return out;
    }

    LegacyKeyStatus legacyKeyStatus(const std::string& key, std::string* reason) {
        const auto& table = statusTable();
        const auto it = table.find(key);
        if (it == table.end()) {
            if (reason) *reason = "not a key SIRIUS recognizes";
            return LegacyKeyStatus::Dropped;
        }
        if (reason) *reason = it->second.reason;
        return it->second.status;
    }

    std::optional<LegacyCropBox> legacyCropBox(const LegacyReconConfig& c) {
        const int lo[3] = {c.cropXmin, c.cropYmin, c.cropZmin};
        const int hi[3] = {c.cropXmax, c.cropYmax, c.cropZmax};
        for (int a = 0; a < 3; ++a)
            if (lo[a] < 0 || hi[a] < lo[a]) return std::nullopt;
        LegacyCropBox b;
        b.x0 = lo[0]; b.nx = hi[0] - lo[0] + 1;
        b.y0 = lo[1]; b.ny = hi[1] - lo[1] + 1;
        b.z0 = lo[2]; b.nz = hi[2] - lo[2] + 1;
        return b;
    }

    std::vector<std::string> legacyConfigKeys() {
        std::vector<std::string> out;
        out.reserve(aliasTable().size());
        for (const auto& entry : aliasTable()) out.push_back(entry.first);
        std::sort(out.begin(), out.end());
        return out;
    }

    LegacyReconConfig loadLegacyConfig(const std::string& path) {
        std::ifstream file(path);
        if (!file)
            throw IoError("Failed to open legacy config: " + path);

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
                throw IoError("Malformed line " + std::to_string(lineNo) +
                              " in " + path + " (expected key=value): " + s);

            const std::string key = trim(s.substr(0, eq));
            const std::string val = trim(s.substr(eq + 1));

            const auto it = table.find(key);
            if (it == table.end())
                throw IoError("Unknown legacy config key '" + key +
                              "' on line " + std::to_string(lineNo) + " of " + path);
            it->second(c, key, val);
            // In file order, once per key: the parser is last-wins, so a
            // repeated key is one entry, at the position it first appeared.
            if (std::find(c.keysPresent.begin(), c.keysPresent.end(), key) == c.keysPresent.end())
                c.keysPresent.push_back(key);
        }
        return c;
    }

    SIMParameters fromLegacy(const LegacyReconConfig& c, LegacyConversionReport* report) {
        SIMParameters p;

        p.ndirs = c.ndirs;
        p.nphases = c.nphases;
        // cudasirecon configs carry nordersout (0: nphases / 2 + 1); a
        // SIRIUS-written norders wins over it. 0 stays "derive".
        p.norders = c.norders > 0 ? c.norders : c.norders_output;
        p.linespacing_um = c.linespacing;
        p.k0_start_angle = c.k0startangle;
        p.na = c.na;
        p.nimm = c.nimm;
        p.wavelength_nm = c.wavelengthNm;
        if (!c.k0angles.empty())
            p.k0_angles = std::vector<double>(c.k0angles.begin(), c.k0angles.end());

        p.dx = c.dxy;
        p.dy = c.dxy;
        p.dz = c.dz;
        p.dz_psf = c.dzPSF;

        p.zoomfact = c.zoomfact;
        p.z_zoom = c.z_zoom;
        p.wiener = c.wiener;
        p.otfcutoff = c.otfcutoff;
        p.background = c.constbkgd;
        p.suppression_radius = c.suppression_radius;
        p.suppress_singularities = (c.bSuppress_singularities != 0);
        p.dampen_order0 = c.bDampenOrder0;
        p.explodefact = c.explodefact;
        p.fast_si = c.bFastSIM;
        p.do_rescale = (c.do_rescale != 0);
        if (!c.phaseSteps.empty())
            p.phase_steps = std::vector<double>(c.phaseSteps.begin(), c.phaseSteps.end());
        if (!c.forceamp.empty())
            p.force_mod_amp = std::vector<double>(c.forceamp.begin(), c.forceamp.end());
        p.equalizez = c.equalizez;
        p.no_kz0 = c.bNoKz0;
        p.filter_overlaps = c.bFilteroverlaps;
        // searchforvector was parsed and stored and then never mapped, so a
        // file saying the pattern vector is known was searched for anyway.
        p.search_pattern_vector = (c.bSearchforvector != 0);

        // Legacy decodes the input apodization from napodize at runtime in
        // apodizationDriver(): >0 => edge ("triangle") blend of that width,
        // exactly -1 => cosine window, anything else (0 or other negatives)
        // => no apodization.
        if (c.napodize > 0) {
            p.apodize_input = ApodizationType::Triangle;
            p.napodize = c.napodize;
        } else if (c.napodize == -1) {
            p.apodize_input = ApodizationType::Cosine;
            p.napodize = 0;            // width is meaningless for the cosine window
        } else {
            p.apodize_input = ApodizationType::None;
            p.napodize = 0;
        }

        switch (c.apodizeoutput) {
            case 0: p.apodize_output = ApodizationType::None; break;
            case 1: p.apodize_output = ApodizationType::Cosine; break;
            case 2: p.apodize_output = ApodizationType::Triangle; break;
            default:
                throw IoError("apodizeoutput must be 0, 1, or 2, got: " +
                              std::to_string(c.apodizeoutput));
        }

        // Filled before validate(), so a caller that catches the validation
        // error still gets the audit of the file it was handed.
        if (report) {
            *report = LegacyConversionReport{};
            for (const std::string& key : c.keysPresent) {
                std::string why;
                switch (legacyKeyStatus(key, &why)) {
                    case LegacyKeyStatus::Applied:
                        report->applied.push_back(key);
                        continue;
                    case LegacyKeyStatus::Pending:
                        report->pending.push_back(key);
                        break;
                    case LegacyKeyStatus::Dropped:
                        report->dropped.push_back(key);
                        break;
                }
                report->notes.push_back(key + ": " + why);
            }
        }

        p.validate();
        return p;
    }

} // namespace sirius
