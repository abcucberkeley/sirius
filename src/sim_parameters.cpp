#include "sirius/sim_parameters.hpp"
#include "sirius/errors.hpp"

#include <toml++/toml.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace sirius {
    namespace {
        std::string_view apodizeToString(ApodizationType a) {
            if (a == ApodizationType::None) return "None";
            if (a == ApodizationType::Cosine) return "Cosine";
            if (a == ApodizationType::Triangle) return "Triangle";
            throw std::runtime_error("Unknown ApodizationType");
        }

        ApodizationType apodizeFromString(std::string_view s) {
            if (s == "None") return ApodizationType::None;
            if (s == "Cosine") return ApodizationType::Cosine;
            if (s == "Triangle") return ApodizationType::Triangle;
            throw std::runtime_error("Unknown apodize_output value: " + std::string(s));
        }
    } // namespace

    double SIMParameters::forcedModAmpFloor(int dir, int order) const noexcept {
        if (!force_mod_amp || force_mod_amp->empty()) return 0.0;
        const std::vector<double>& f = *force_mod_amp;
        // cudasirecon gates the whole feature on `forceamp[0] > 0.0`, with
        // forceamp defaulting to a single 0. Kept exactly: the only case where
        // a per-entry rule would differ is a leading entry <= 0 followed by a
        // positive one, and there cudasirecon forces nothing.
        if (f[0] <= 0.0) return 0.0;
        const int orders = resolvedOrders();
        // Order 0 is the widefield band. cudasirecon's loop starts at 1 and
        // SIRIUS overwriting amps[0] with a real constant is what made the
        // fitted amplitudes of findings 9k.51 incomparable.
        if (order < 1 || order >= orders) return 0.0;
        if (dir < 0 || dir >= ndirs) return 0.0;

        const std::size_t n = f.size();
        const std::size_t d = static_cast<std::size_t>(dir);
        const std::size_t o = static_cast<std::size_t>(order);
        const std::size_t all = static_cast<std::size_t>(orders);
        const std::size_t side = static_cast<std::size_t>(orders - 1);
        const std::size_t dirs = static_cast<std::size_t>(ndirs);

        // SIRIUS's two older lengths are tested FIRST, so that a length which
        // already validated today keeps the meaning it had today: at
        // ndirs == 2, norders == 2 a list of 2 is ndirs*(norders-1) as well as
        // norders, and it was norders before this function existed.
        std::size_t idx = 0;
        if (n == all)                  idx = o;              // shared, order-indexed
        else if (n == dirs * all)      idx = d * all + o;    // per direction, order-indexed
        else if (n == side)            idx = o - 1;          // shared, side bands (cudasirecon's)
        else if (n == dirs * side)     idx = d * side + o - 1;
        else return 0.0;                                     // validate() refuses these
        return f[idx];
    }

    void SIMParameters::validate() const {
        // NaN passes every range check below (each comparison with it is
        // false), and an infinity overflows the cutoffs derived from these.
        const std::pair<const char*, double> reals[] = {
            {"k0_start_angle", k0_start_angle}, {"linespacing_um", linespacing_um}, {"na", na}, {"nimm", nimm}, {"wavelength_nm", wavelength_nm}, {"dx", dx}, {"dy", dy}, {"dz", dz}, {"dz_psf", dz_psf}, {"zoomfact", zoomfact}, {"wiener", wiener}, {"otfcutoff", otfcutoff}, {"background", background}, {"explodefact", explodefact}};
        for (const auto& [name, value] : reals)
            if (!std::isfinite(value)) throw std::runtime_error(std::string(name) + " must be finite");
        if (k0_angles)
            for (double a : *k0_angles)
                if (!std::isfinite(a)) throw std::runtime_error("k0_angles must be finite");

        if (ndirs < 1) throw std::runtime_error("ndirs must be >= 1");
        if (nphases < 1) throw std::runtime_error("nphases must be >= 1");
        if (norders < 0) throw std::runtime_error("norders must be >= 0 (0 derives nphases / 2 + 1)");
        // The widefield order alone is no SIM (and the k0 fit would divide by
        // its order 0), and 2 * orders - 1 bands need as many phases.
        const int orders = resolvedOrders();
        if (orders < 2)
            throw std::runtime_error("at least 2 orders are needed, got " + std::to_string(orders) +
                                     (norders > 0 ? " (norders)" : " (nphases / 2 + 1)"));
        if (nphases < 2 * orders - 1)
            throw std::runtime_error(std::to_string(nphases) + " phases cannot separate " + std::to_string(orders) +
                                     " orders (that needs " + std::to_string(2 * orders - 1) + " phases)");
        if (linespacing_um <= 0.0) throw std::runtime_error("linespacing_um must be > 0");
        if (k0_angles && static_cast<int>(k0_angles->size()) != ndirs)
            throw std::runtime_error("k0_angles size must equal ndirs");
        if (phase_steps) {
            for (double a : *phase_steps)
                if (!std::isfinite(a)) throw std::runtime_error("phase_steps must be finite");
            if (static_cast<int>(phase_steps->size()) != nphases)
                throw std::runtime_error("phase_steps length must equal nphases (" + std::to_string(nphases) + "), got " +
                                         std::to_string(phase_steps->size()));
        }
        if (force_mod_amp) {
            for (double a : *force_mod_amp)
                if (!std::isfinite(a)) throw std::runtime_error("force_mod_amp must be finite");
            const int n = static_cast<int>(force_mod_amp->size());
            // cudasirecon indexes forceamp[order - 1] over order = 1..norders-1
            // and shares one list across directions, so norders-1 is its own
            // length -- and the single 0.5 of a 3-phase config is that length.
            // Demanding `orders` here is what refused the user's own file.
            const int side = orders - 1;
            if (n != orders && n != ndirs * orders && n != side && n != ndirs * side)
                throw std::runtime_error("force_mod_amp length must be norders-1 (" + std::to_string(side) +
                                         "), ndirs*(norders-1) (" + std::to_string(ndirs * side) + "), norders (" +
                                         std::to_string(orders) + ") or ndirs*norders (" +
                                         std::to_string(ndirs * orders) + "), got " + std::to_string(n));
        }
        if (na <= 0.0) throw std::runtime_error("na must be > 0");
        if (nimm <= 0.0) throw std::runtime_error("nimm must be > 0");
        // The reconstruction takes asin(na / nimm) for the OTF's axial
        // support: beyond 1 that is NaN, which became an INT_MIN plane index
        // and a write far outside the band storage. na == nimm is a
        // 90-degree aperture and still reconstructs.
        if (na > nimm)
            throw std::runtime_error("na " + std::to_string(na) + " must not exceed the immersion index nimm " +
                                     std::to_string(nimm));
        if (wavelength_nm <= 0.0) throw std::runtime_error("wavelength_nm must be > 0");
        if (dx <= 0.0) throw std::runtime_error("dx must be > 0");
        if (dy <= 0.0) throw std::runtime_error("dy must be > 0");
        if (dz <= 0.0) throw std::runtime_error("dz must be > 0");
        if (dz_psf <= 0.0) throw std::runtime_error("dz_psf must be > 0");
        // the assembly writes every input frequency into the output grid
        if (zoomfact < 1.0) throw std::runtime_error("zoomfact must be >= 1 (the output grid cannot be smaller than the input)");
        if (z_zoom < 1) throw std::runtime_error("z_zoom must be >= 1");
        if (wiener < 0.0) throw std::runtime_error("wiener must be >= 0");
        if (otfcutoff < 0.0) throw std::runtime_error("otfcutoff must be >= 0");
        if (napodize < 0) throw std::runtime_error("napodize must be >= 0");
        if (suppression_radius < 0) throw std::runtime_error("suppression_radius must be >= 0");
        if (explodefact <= 0.0) throw std::runtime_error("explodefact must be > 0");
    }

    void saveParameters(const std::string& path, const SIMParameters& p) {
        p.validate();

        toml::table optics;
        optics.insert("ndirs", p.ndirs);
        optics.insert("nphases", p.nphases);
        optics.insert("norders", p.norders);
        optics.insert("linespacing_um", p.linespacing_um);
        optics.insert("k0_start_angle", p.k0_start_angle);
        optics.insert("na", p.na);
        optics.insert("nimm", p.nimm);
        optics.insert("wavelength_nm", p.wavelength_nm);
        // A property of the illumination, so it lives with the optics rather
        // than with the output knobs. Only idealOTF reads it.
        optics.insert("illumination_has_axial_component", p.illumination_has_axial_component);
        if (p.k0_angles) {
            toml::array arr;
            for (double a : *p.k0_angles)
                arr.push_back(a);
            optics.insert("k0_angles", std::move(arr));
        }
        auto writeDoubles = [](toml::table& table, const char* key, const std::optional<std::vector<double>>& values) {
            if (!values) return;
            toml::array arr;
            for (double a : *values) arr.push_back(a);
            table.insert(key, std::move(arr));
        };
        writeDoubles(optics, "phase_steps", p.phase_steps);
        writeDoubles(optics, "force_mod_amp", p.force_mod_amp);

        toml::table pixels;
        pixels.insert("dx", p.dx);
        pixels.insert("dy", p.dy);
        pixels.insert("dz", p.dz);
        pixels.insert("dz_psf", p.dz_psf);

        toml::table output;
        output.insert("zoomfact", p.zoomfact);
        output.insert("z_zoom", p.z_zoom);
        output.insert("wiener", p.wiener);
        output.insert("otfcutoff", p.otfcutoff);
        output.insert("background", p.background);
        output.insert("apodize_input", std::string(apodizeToString(p.apodize_input)));
        output.insert("napodize", p.napodize);
        output.insert("suppression_radius", p.suppression_radius);
        output.insert("suppress_singularities", p.suppress_singularities);
        output.insert("dampen_order0", p.dampen_order0);
        output.insert("apodize_output", std::string(apodizeToString(p.apodize_output)));
        output.insert("explodefact", p.explodefact);
        output.insert("fast_si", p.fast_si);
        output.insert("do_rescale", p.do_rescale);
        output.insert("equalizez", p.equalizez);
        output.insert("no_kz0", p.no_kz0);
        output.insert("filter_overlaps", p.filter_overlaps);
        output.insert("search_pattern_vector", p.search_pattern_vector);

        toml::table tbl;
        tbl.insert("optics", std::move(optics));
        tbl.insert("pixels", std::move(pixels));
        tbl.insert("output", std::move(output));

        std::ofstream file(path);
        if (!file)
            throw IoError("Failed to open for writing: " + path);
        file << tbl;
    }

    SIMParameters loadParameters(const std::string& path) {
        toml::table tbl;
        try {
            tbl = toml::parse_file(path);
        } catch (const toml::parse_error& e) {
            throw IoError(std::string("Failed to parse config: ") + e.what());
        }

        SIMParameters p;  // starts from defaults

        auto optics = tbl["optics"];
        p.ndirs = optics["ndirs"].value_or(p.ndirs);
        p.nphases = optics["nphases"].value_or(p.nphases);
        p.norders = optics["norders"].value_or(p.norders);
        p.linespacing_um = optics["linespacing_um"].value_or(p.linespacing_um);
        p.k0_start_angle = optics["k0_start_angle"].value_or(p.k0_start_angle);
        p.na = optics["na"].value_or(p.na);
        p.nimm = optics["nimm"].value_or(p.nimm);
        p.wavelength_nm = optics["wavelength_nm"].value_or(p.wavelength_nm);
        p.illumination_has_axial_component =
            optics["illumination_has_axial_component"].value_or(p.illumination_has_axial_component);

        auto readDoubles = [](auto node) -> std::optional<std::vector<double>> {
            auto* arr = node.as_array();
            if (!arr) return std::nullopt;
            std::vector<double> values;
            values.reserve(arr->size());
            for (auto& item : *arr)
                if (auto v = item.template value<double>()) values.push_back(*v);
            if (values.empty()) return std::nullopt;
            return values;
        };
        if (auto angles = readDoubles(optics["k0_angles"])) p.k0_angles = std::move(angles);
        if (auto phases = readDoubles(optics["phase_steps"])) p.phase_steps = std::move(phases);
        if (auto amps = readDoubles(optics["force_mod_amp"])) p.force_mod_amp = std::move(amps);

        auto pixels = tbl["pixels"];
        p.dx = pixels["dx"].value_or(p.dx);
        p.dy = pixels["dy"].value_or(p.dy);
        p.dz = pixels["dz"].value_or(p.dz);
        p.dz_psf = pixels["dz_psf"].value_or(p.dz_psf);

        auto output = tbl["output"];
        p.zoomfact = output["zoomfact"].value_or(p.zoomfact);
        p.z_zoom = output["z_zoom"].value_or(p.z_zoom);
        p.wiener = output["wiener"].value_or(p.wiener);
        p.otfcutoff = output["otfcutoff"].value_or(p.otfcutoff);
        p.background = output["background"].value_or(p.background);
        p.napodize = output["napodize"].value_or(p.napodize);
        p.suppression_radius = output["suppression_radius"].value_or(p.suppression_radius);
        p.suppress_singularities = output["suppress_singularities"].value_or(p.suppress_singularities);
        p.dampen_order0 = output["dampen_order0"].value_or(p.dampen_order0);
        p.explodefact = output["explodefact"].value_or(p.explodefact);
        p.fast_si = output["fast_si"].value_or(p.fast_si);
        p.do_rescale = output["do_rescale"].value_or(p.do_rescale);
        p.equalizez = output["equalizez"].value_or(p.equalizez);
        p.no_kz0 = output["no_kz0"].value_or(p.no_kz0);
        p.filter_overlaps = output["filter_overlaps"].value_or(p.filter_overlaps);
        p.search_pattern_vector = output["search_pattern_vector"].value_or(p.search_pattern_vector);

        if (auto s = output["apodize_input"].value<std::string>())
            p.apodize_input = apodizeFromString(*s);
        if (auto s = output["apodize_output"].value<std::string>())
            p.apodize_output = apodizeFromString(*s);

        p.validate();
        return p;
    }

} // namespace sirius