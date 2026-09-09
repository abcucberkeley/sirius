#ifndef SIRIUS_APP_PIPELINE_HPP
#define SIRIUS_APP_PIPELINE_HPP

// The ordered stack of steps. Step 0 is always the pinned Load step; any
// order of the others is legal (validation is per operation). Steps carry
// their own id so cached outputs and undo entries survive reordering.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "core/operation.hpp"
#include "core/params.hpp"

namespace sirius::app {

    using StepId = std::uint64_t;

    struct Step {
        StepId id = 0;
        std::string kind;
        std::string name;                 // editable display name
        bool enabled = true;
        bool pinned = false;              // the Load step
        CachePolicy cache = CachePolicy::Recompute;
        ParamSet params;

        const Operation& op() const { return requireOperation(kind); }
        // "01", "02", ...
        static std::string number(int index);
    };

    class Pipeline {
    public:
        Pipeline();                                    // Load step only

        const std::vector<Step>& steps() const noexcept { return steps_; }
        int size() const noexcept { return static_cast<int>(steps_.size()); }
        const Step& at(int index) const { return steps_.at(static_cast<std::size_t>(index)); }
        Step& at(int index) { return steps_.at(static_cast<std::size_t>(index)); }
        int indexOf(StepId id) const noexcept;        // -1 when unknown
        const Step* find(StepId id) const noexcept;
        Step* find(StepId id) noexcept;
        int enabledCount() const noexcept;

        // Appends (or inserts at `at` >= 1) a step of `kind` with its default
        // parameters; returns its id. Throws for an unknown kind.
        StepId add(const std::string& kind, int at = -1);
        StepId insertStep(Step step, int at = -1);    // step.id assigned when 0
        void remove(int index);                       // no-op for the Load step
        bool move(int index, int delta);              // false when the move is illegal
        StepId duplicate(int index);
        void setEnabled(int index, bool on);          // Load stays enabled
        void setParams(int index, ParamSet params);
        void setCache(int index, CachePolicy policy);
        void rename(int index, std::string name);
        void clearAfterLoad();

        // Replace everything but keep the Load step's parameters when `keepLoad`.
        void replaceSteps(std::vector<Step> steps, bool keepLoad);

        // The id the next added step gets, and a floor for it: a pipeline
        // rebuilt from a snapshot (undo) must not hand out the ids of steps
        // the snapshot no longer holds -- their cached outputs still exist.
        StepId peekNextId() const noexcept { return nextId_; }
        void reserveIds(StepId next) noexcept { nextId_ = std::max(nextId_, next); }

        nlohmann::json toJson() const;
        static Pipeline fromJson(const nlohmann::json& j);
        // TOML on disk (".sirius.toml").
        void save(const std::string& path) const;
        static Pipeline load(const std::string& path);
        // Python script that reproduces the pipeline with the sirius package.
        std::string toPythonScript(const std::string& datasetPath) const;

        // The example pipeline of the design (SIM → einsum → contrast → merge → segment → volume).
        static Pipeline example();

    private:
        StepId nextId();
        std::vector<Step> steps_;
        StepId nextId_ = 1;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_PIPELINE_HPP
