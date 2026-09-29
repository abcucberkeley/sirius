#ifndef SIRIUS_APP_PARAMS_HPP
#define SIRIUS_APP_PARAMS_HPP

// Typed, ordered parameter sets. Every operation declares its parameters as
// ParamSpecs (key, type, default, range, choices); the generic form in the
// parameters dock, the pipeline file format, the Python export and the
// assistant's tool schema are all generated from those specs, so adding a
// parameter to an operation is one line.

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace sirius::app {

    using ParamValue = std::variant<bool, std::int64_t, double, std::string, std::vector<double>,
                                    std::vector<std::string>>;

    enum class ParamType {
        Bool,
        Int,
        Double,
        String,
        Path,        // file or directory; `fileFilter` / `directory` refine it
        Choice,      // one of `choices`
        Channel,     // index into the input's channels (int)
        Axes,        // subset of "ctzyx" (string)
        DoubleList,  // e.g. a (z, y, x) triple
        StringList,
    };

    class ParamSet;   // ParamSpec::visibleFor asks it for the controlling value

    struct ParamSpec {
        std::string key;
        std::string label;
        ParamType type = ParamType::Double;
        ParamValue defaultValue = 0.0;
        double min = -std::numeric_limits<double>::infinity();
        double max = std::numeric_limits<double>::infinity();
        double step = 0.0;                     // spin box step; 0 = automatic
        int decimals = -1;                     // -1 = automatic
        std::vector<std::string> choices;      // ParamType::Choice
        std::string unit;                      // "µm", "px", "%"
        std::string help;                      // tooltip / help table body
        std::string fileFilter;                // "TIFF (*.tif *.tiff)"
        bool directory = false;                // Path picks a directory
        bool advanced = false;                 // hidden behind "More"
        bool readOnly = false;                 // shown, not editable (facts)
        std::string group;                     // optional section caption

        // Shown only while other parameters have (or have not) certain values:
        // a mode that reads its settings from a file has no use for the fields
        // it overrides, and a panel that shows them anyway invites the user to
        // change something that will be ignored. Every rule has to hold; no
        // rules means always shown. Purely a display rule -- the value is still
        // stored, still saved and still read, so turning the mode back reveals
        // it unchanged.
        struct Visibility {
            std::string key;
            std::vector<std::string> values;
            bool negate = false;
        };
        std::vector<Visibility> visibility;

        // Fluent helpers so op tables read as one line per parameter.
        ParamSpec& range(double lo, double hi, double st = 0.0, int dec = -1) {
            min = lo;
            max = hi;
            step = st;
            decimals = dec;
            return *this;
        }
        ParamSpec& withUnit(std::string u) {
            unit = std::move(u);
            return *this;
        }
        ParamSpec& withHelp(std::string h) {
            help = std::move(h);
            return *this;
        }
        ParamSpec& withChoices(std::vector<std::string> c) {
            choices = std::move(c);
            return *this;
        }
        ParamSpec& withFilter(std::string f) {
            fileFilter = std::move(f);
            return *this;
        }
        ParamSpec& asDirectory() {
            directory = true;
            return *this;
        }
        ParamSpec& asAdvanced() {
            advanced = true;
            return *this;
        }
        ParamSpec& asReadOnly() {
            readOnly = true;
            return *this;
        }
        ParamSpec& inGroup(std::string g) {
            group = std::move(g);
            return *this;
        }
        ParamSpec& visibleWhen(std::string otherKey, std::vector<std::string> values) {
            visibility.push_back({std::move(otherKey), std::move(values), false});
            return *this;
        }
        ParamSpec& hiddenWhen(std::string otherKey, std::vector<std::string> values) {
            visibility.push_back({std::move(otherKey), std::move(values), true});
            return *this;
        }
        // Whether this parameter should be shown for the given settings.
        bool visibleFor(const ParamSet& p) const;
    };

    ParamSpec boolParam(std::string key, std::string label, bool def);
    ParamSpec intParam(std::string key, std::string label, std::int64_t def);
    ParamSpec doubleParam(std::string key, std::string label, double def);
    ParamSpec stringParam(std::string key, std::string label, std::string def);
    ParamSpec pathParam(std::string key, std::string label, std::string def = {});
    ParamSpec choiceParam(std::string key, std::string label, std::vector<std::string> choices, std::string def);
    ParamSpec channelParam(std::string key, std::string label, std::int64_t def = 0);
    ParamSpec axesParam(std::string key, std::string label, std::string def);
    ParamSpec doubleListParam(std::string key, std::string label, std::vector<double> def);

    class ParamSet {
    public:
        ParamSet() = default;
        // Defaults of every spec.
        explicit ParamSet(const std::vector<ParamSpec>& specs);

        bool has(const std::string& key) const noexcept;
        const ParamValue* find(const std::string& key) const noexcept;
        void set(const std::string& key, ParamValue value);   // appends when new
        void erase(const std::string& key);
        std::size_t size() const noexcept { return items_.size(); }
        const std::vector<std::pair<std::string, ParamValue>>& items() const noexcept { return items_; }

        // Typed getters with lenient conversion (int <-> double, "3" -> 3).
        bool getBool(const std::string& key, bool def = false) const;
        std::int64_t getInt(const std::string& key, std::int64_t def = 0) const;
        double getDouble(const std::string& key, double def = 0.0) const;
        std::string getString(const std::string& key, std::string def = {}) const;
        std::vector<double> getDoubleList(const std::string& key) const;
        std::vector<std::string> getStringList(const std::string& key) const;

        // Fill missing keys from the specs' defaults; drop unknown ones when `strict`.
        void applyDefaults(const std::vector<ParamSpec>& specs, bool strict = false);
        // Clamp numeric values into their spec range and coerce types.
        void coerce(const std::vector<ParamSpec>& specs);

        nlohmann::json toJson() const;
        static ParamSet fromJson(const nlohmann::json& j);

        friend bool operator==(const ParamSet& a, const ParamSet& b) noexcept { return a.items_ == b.items_; }
        friend bool operator!=(const ParamSet& a, const ParamSet& b) noexcept { return !(a == b); }

    private:
        std::vector<std::pair<std::string, ParamValue>> items_;
    };

    nlohmann::json toJson(const ParamValue& v);
    ParamValue paramValueFromJson(const nlohmann::json& j);
    // Human readable: "0.001", "Cosine", "z, y, x", "on"
    std::string toDisplayString(const ParamValue& v);
    // Coerce a JSON value (from the assistant or a file) into the spec's type;
    // throws std::invalid_argument with a message naming the key.
    ParamValue coerceToSpec(const ParamSpec& spec, const nlohmann::json& j);
    // JSON-schema fragment describing the spec (for the assistant tools).
    nlohmann::json schemaOf(const ParamSpec& spec);

} // namespace sirius::app

#endif // SIRIUS_APP_PARAMS_HPP
