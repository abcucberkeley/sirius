#ifndef SIRIUS_APP_HISTORY_HPP
#define SIRIUS_APP_HISTORY_HPP

// Undo / redo as a stack of reversible commands. The workbench builds
// commands from closures that capture "before" and "after" state (pipeline
// JSON, view state, label diffs), so any edit -- from the UI, the assistant
// or a script -- is undone the same way.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sirius::app {

    struct Command {
        std::string label;              // "Set Wiener 0.001 → 0.002"
        std::function<void()> undo;
        std::function<void()> redo;
        // Consecutive commands with the same non-empty merge key collapse into
        // one entry (slider drags, brush strokes): the newest command's
        // closures replace the entry's, so the caller composes them to span
        // the whole merged range (see Workbench::pushEdit). Any other push,
        // and an undo, ends the group: the history is the single source of
        // truth for what merges (mergesWith).
        std::string mergeKey;
    };

    class History {
    public:
        void push(Command c);            // clears the redo stack
        // True when a command with this key would merge into the top undo
        // entry (same non-empty key, and no other push, undo or redo since):
        // what the caller checks to compose the "before" state of a merged
        // group.
        bool mergesWith(const std::string& key) const noexcept;
        bool canUndo() const noexcept { return !undo_.empty(); }
        bool canRedo() const noexcept { return !redo_.empty(); }
        std::string undoLabel() const;   // "" when nothing
        std::string redoLabel() const;
        void undo();
        void redo();
        void clear();
        std::size_t size() const noexcept { return undo_.size(); }
        void setLimit(std::size_t n) noexcept { limit_ = n; }
        // The state the history stands at, as a number: every push -- a
        // merge included -- makes a new one, undo and redo return to the
        // number that state had, and clear() keeps it (forgetting how the
        // state came about does not change it). Compare it with the number
        // recorded at a save to tell whether anything changed since; the
        // entry count cannot (a merged drag, or an undo and a new edit,
        // leave it as it was).
        std::uint64_t revision() const noexcept { return undo_.empty() ? base_ : undo_.back().revision; }

    private:
        struct Entry {
            Command command;
            std::uint64_t revision = 0;  // the state after the command
        };
        std::vector<Entry> undo_, redo_;
        bool mergeOpen_ = false;         // the top entry is still taking merges
        std::size_t limit_ = 200;
        std::uint64_t base_ = 0;         // the state before the oldest undo entry
        std::uint64_t lastRevision_ = 0;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_HISTORY_HPP
