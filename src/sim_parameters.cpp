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
        if (p.k0_angles) {
            toml::array arr;
            for (double a : *p.k0_angles)
                arr.push_back(a);
            optics.insert("k0_angles", std::move(arr));
        }

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

        if (auto* arr = optics["k0_angles"].as_array()) {
            std::vector<double> angles;
            angles.reserve(arr->size());
            for (auto& node : *arr) {
                if (auto v = node.value<double>())
                    angles.push_back(*v);
            }
            if (!angles.empty())
                p.k0_angles = std::move(angles);
        }

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

        if (auto s = output["apodize_input"].value<std::string>())
            p.apodize_input = apodizeFromString(*s);
        if (auto s = output["apodize_output"].value<std::string>())
            p.apodize_output = apodizeFromString(*s);

        p.validate();
        return p;
    }

} // namespace sirius